import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from utils import *
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Choose your model(s) & language(s) (vLLM inference)')
parser.add_argument('--model', type=str,
                    help='Model name or path. Multiple models: comma-separated.')
parser.add_argument('--language', type=str, default=None,
                    help='Language(s), comma-separated. Must pair 1:1 with --country.')
parser.add_argument('--country', type=str, default=None,
                    help='Country/ies, comma-separated. Must pair 1:1 with --language.')
parser.add_argument('--question_dir', type=str, default=None,
                    help='Provide the directory name with (translated) questions.')
parser.add_argument('--question_file', type=str, default=None,
                    help='Provide the csv file name with (translated) questions.')
parser.add_argument('--question_col', type=str, default=None,
                    help='Override question column name. Auto-detected from language/country if omitted.')
parser.add_argument('--prompt_dir', type=str, default=None,
                    help='Provide the directory where the prompts are saved.')
parser.add_argument('--prompt_file', type=str, default=None,
                    help='Provide the name of the csv file where the prompts are saved.')
parser.add_argument('--prompt_no', type=str, default=None,
                    help='Prompt ID(s), comma-separated (e.g. "inst-4,pers-3").')
parser.add_argument('--id_col', type=str, default="ID",
                    help='Column name for question IDs.')
parser.add_argument('--output_dir', type=str, default='./model_inference_results_vllm',
                    help='Directory for output files.')
parser.add_argument('--output_file', type=str, default=None,
                    help='(Unused; output filenames are derived from model/country/language/prompt_no.)')
parser.add_argument('--model_cache_dir', type=str, default='/home/ec2-user/efs/huggingface',
                    help='Directory for model cache.')
parser.add_argument('--temperature', type=int, default=0,
                    help='Generation temperature.')
parser.add_argument('--top_p', type=int, default=1,
                    help='Generation top_p.')
parser.add_argument('--max_length', type=int, default=512,
                    help='Max total sequence length (prompt + generated tokens).')
parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing output files instead of skipping them.')

args = parser.parse_args()


def make_prompt(question, prompt_no, language, _country, prompt_sheet):
    prompt = prompt_sheet[prompt_sheet['id'] == prompt_no]
    if language == 'English':
        prompt = prompt['English'].values[0]
    else:
        prompt = prompt['Translation'].values[0]
    return prompt.replace('{q}', question)


def format_prompt_for_model(prompt, model_path, tokenizer):
    """Apply chat template for models that use it; pass raw prompt otherwise."""
    if any(key in model_path for key in ['Orion', 'Qwen', 'SeaLLM', 'Merak']) or \
       any(key in model_path.lower() for key in ['mistral', 'c4ai', 'aya-23']):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return prompt


def get_response_from_all():
    def split_arg(s):
        """Split a comma-separated CLI arg into a list; returns [None] if s is None."""
        if s is None:
            return [None]
        return [x.strip() for x in s.split(',')]

    models    = split_arg(args.model)
    languages = split_arg(args.language)
    countries = split_arg(args.country)
    prompt_nos = split_arg(args.prompt_no)

    question_dir = args.question_dir
    question_col = args.question_col
    id_col       = args.id_col
    output_dir   = os.path.join(args.output_dir, os.path.basename(args.model))
    overwrite    = args.overwrite

    os.makedirs(output_dir, exist_ok=True)

    if len(languages) != len(countries):
        print("ERROR: --language and --country must have the same number of entries.")
        exit(1)

    def get_questions(country):
        return pd.read_csv(os.path.join(question_dir, f'{country}_questions.csv'), encoding='utf-8')

    for model_name in models:
        model_path = MODEL_PATHS[model_name] if model_name in MODEL_PATHS else model_name

        # ------------------------------------------------------------------
        # Load tokenizer and vLLM model ONCE per model
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"Model: {model_path}")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=os.path.join(args.model_cache_dir, model_path),
        )
        llm = LLM(
            model=model_path,
            download_dir=args.model_cache_dir,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            tensor_parallel_size=4,
        )

        # ------------------------------------------------------------------
        # Phase 1: Compile ALL prompts across every (language, country, prompt_no)
        # ------------------------------------------------------------------
        # all_items: list of (output_filename, guid, q, prompt, formatted_prompt, pno)
        all_items = []
        # files_to_init: output_filename -> (id_col, q_col) for header writing
        files_to_init = {}
        # Cache prompt sheets per country to avoid redundant disk reads
        prompt_sheets = {}

        for lang, country in zip(languages, countries):
            replace_country_flag = (lang == 'English' and COUNTRY_LANG.get(country) != 'English')

            # Auto-detect question column if not explicitly provided
            if question_col is not None:
                q_col = question_col
            elif lang == COUNTRY_LANG.get(country):
                q_col = 'Translation'
            elif lang == 'English':
                q_col = 'Question'
            else:
                q_col = 'Translation'

            questions_df = get_questions(country)
            print(f"\n[{country} / {lang}]  q_col={q_col}  n={len(questions_df)}")

            for pno in prompt_nos:
                if pno is not None:
                    output_filename = os.path.join(
                        output_dir,
                        f"{os.path.basename(model_path)}-{country}_{lang}_{pno}_result.csv",
                    )
                    if country not in prompt_sheets:
                        prompt_sheets[country] = pd.read_csv(
                            os.path.join('data/prompts', f'{country}_prompts.csv')
                        )
                    prompt_sheet = prompt_sheets[country]
                else:
                    output_filename = os.path.join(
                        output_dir,
                        f"{os.path.basename(model_path)}-{country}_{lang}_result.csv",
                    )
                    prompt_sheet = None

                # Skip if output already exists and --overwrite not set
                if os.path.exists(output_filename) and not overwrite:
                    print(f"  Skipping {os.path.basename(output_filename)} (exists; use --overwrite to force)")
                    continue

                # Register this file for header initialization
                if output_filename not in files_to_init:
                    files_to_init[output_filename] = (id_col, q_col)

                for _, d in questions_df.iterrows():
                    q    = d[q_col]
                    guid = d[id_col]

                    if replace_country_flag:
                        q = replace_country_name(q, country.replace('_', ' '))

                    if pno is not None:
                        prompt = make_prompt(q, pno, lang, country, prompt_sheet)
                    else:
                        prompt = q

                    formatted_prompt = format_prompt_for_model(prompt, model_path, tokenizer)
                    all_items.append((output_filename, guid, q, prompt, formatted_prompt, pno))

        if not all_items:
            print(f"\nNo prompts to process for model {model_name} (all output files already exist).")
            continue

        print(f"\n{'='*60}")
        print(f"Total prompts compiled : {len(all_items)}")
        print(f"Output files           : {len(files_to_init)}")
        print(f"{'='*60}")

        # Initialize output files (create fresh; delete first if overwriting)
        for output_filename, (f_id_col, f_q_col) in files_to_init.items():
            if os.path.exists(output_filename) and overwrite:
                os.remove(output_filename)
            write_csv_row([f_id_col, f_q_col, 'prompt', 'response', 'prompt_no'], output_filename)

        # ------------------------------------------------------------------
        # Phase 2: Compute per-prompt max_tokens (total_length - prompt_length)
        # ------------------------------------------------------------------
        print("\nTokenizing prompts...")
        formatted_prompts = [item[4] for item in all_items]
        sampling_params_list = []
        for fp in formatted_prompts:
            n_prompt_tokens = len(tokenizer.encode(fp))
            mt = max(1, args.max_length - n_prompt_tokens)
            sampling_params_list.append(SamplingParams(temperature=0, max_tokens=mt))

        # ------------------------------------------------------------------
        # Phase 3: Single vLLM inference call for ALL prompts
        # ------------------------------------------------------------------
        print(f"\nRunning vLLM inference on {len(all_items)} prompts...")
        outputs = llm.generate(formatted_prompts, sampling_params_list)

        # ------------------------------------------------------------------
        # Phase 4: Write results to their respective output files
        # ------------------------------------------------------------------
        print("\nWriting results...")
        for (output_filename, guid, q, prompt, _, pno), output in zip(all_items, outputs):
            response = output.outputs[0].text.strip()
            print(f"ID: {guid} | File: {os.path.basename(output_filename)}")
            print(f"  Prompt  : {prompt[:120]}")
            print(f"  Response: {response[:120]}")
            write_csv_row([guid, q, prompt, response, pno], output_filename)

        print(f"\nFinished model: {model_name}")


if __name__ == "__main__":
    get_response_from_all()
