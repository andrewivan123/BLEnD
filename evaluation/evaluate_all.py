from evaluation_utils import *
from exact_match import *
from itertools import product

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate all models, countries, and languages in a single pass.')
    parser.add_argument('--models', nargs='+', required=True,
                        help='One or more model keys/paths to evaluate.')
    parser.add_argument('--country_lang', nargs='+', required=True,
                        help='Country-language pairs in "Country:Language" format (e.g. "UK:English" "South_Korea:Korean").')
    parser.add_argument('--prompt_nos', nargs='+', required=True,
                        help='One or more prompt IDs (e.g. inst-4 pers-3).')
    parser.add_argument('--id_col', type=str, default=None,
                        help='Column name with question IDs in the response CSV.')
    parser.add_argument('--question_col', type=str, default=None,
                        help='Column name with questions in the response CSV.')
    parser.add_argument('--response_col', type=str, default=None,
                        help='Column name with LLM responses in the response CSV.')
    parser.add_argument('--response_dir', type=str, default='../model_inference_results',
                        help='Directory containing model response files.')
    parser.add_argument('--annotation_dir', type=str, default='../final_dataset',
                        help='Directory containing annotation files.')
    parser.add_argument('--annotation_filename', type=str, default='{country}_data.json',
                        help='Annotation filename template (use {country} as placeholder).')
    parser.add_argument('--annotations_key', type=str, default='annotations',
                        help='Key for annotations within the annotation file.')
    parser.add_argument('--evaluation_result_file', type=str, default='evaluation_results.csv',
                        help='Output CSV for evaluation results.')
    parser.add_argument('--skip_mcq', action='store_true',
                        help='Skip multiple choice question evaluation.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-run evaluation even if results already exist in the output CSV.')

    args = parser.parse_args()

    # Build full run list (native language + English sub-run for non-English countries)
    country_lang_pairs = [cl.split(':', 1) for cl in args.country_lang]
    runs = []
    for model, (country, language), prompt_no in product(args.models, country_lang_pairs, args.prompt_nos):
        runs.append((model, country, language, prompt_no))
        if language != 'English':
            runs.append((model, country, 'English', prompt_no))

    # Filter out already-done runs with a single CSV read upfront
    if not args.overwrite and os.path.exists(args.evaluation_result_file):
        existing = pd.read_csv(args.evaluation_result_file)
        done_keys = set(zip(
            existing['model'], existing['country'], existing['language'], existing['prompt_no']
        ))
        pending, skipped = [], []
        for run in runs:
            (pending if run not in done_keys else skipped).append(run)
        for model, country, language, prompt_no in skipped:
            print(f"[skip] {model} | {country} | {language} | {prompt_no}")
        runs = pending

    # Create CSV with header if it doesn't exist yet
    if not os.path.exists(args.evaluation_result_file):
        write_csv_row(['model', 'country', 'language', 'prompt_no', 'eval_method', 'score'],
                      args.evaluation_result_file)

    total = len(runs)
    for i, (model, country, language, prompt_no) in enumerate(runs, 1):
        print(f"\n[{i}/{total}] model={model}  country={country}  language={language}  prompt={prompt_no}")

        res_df = get_model_response_file(
            data_dir=args.response_dir, model=model, country=country,
            language=language, prompt_no=prompt_no
        )
        real_annotation = get_annotations(
            data_dir=args.annotation_dir, country=country, template=args.annotation_filename
        )

        sem_b, sem_w, res_df = soft_exact_match(
            country=country, language=language,
            annotation_dict=real_annotation, response_df=res_df,
            id_col=args.id_col, r_col=args.response_col,
            annotations_key=args.annotations_key
        )

        write_csv_row([model, country, language, prompt_no, 'SEM-B', sem_b], args.evaluation_result_file)
        write_csv_row([model, country, language, prompt_no, 'SEM-W', sem_w], args.evaluation_result_file)

        score_path = os.path.join(
            args.response_dir, os.path.basename(model),
            f'{os.path.basename(model)}_{country}_{language}_{prompt_no}_response_score.csv'
        )
        res_df.to_csv(score_path, index=False, encoding='utf-8')

    # Final dedup pass over the CSV
    if os.path.exists(args.evaluation_result_file):
        df = pd.read_csv(args.evaluation_result_file)
        df.drop_duplicates(
            subset=['model', 'country', 'language', 'prompt_no', 'eval_method'],
            keep='last', inplace=True
        )
        df.to_csv(args.evaluation_result_file, index=False, encoding='utf-8')
