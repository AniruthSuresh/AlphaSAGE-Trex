import json
import argparse
import os

def extract_top_alphas(args):
    print(f"Loading {args.file}...")
    with open(args.file, 'r') as f:
        data = json.load(f)

    extracted_exprs = []
    
    if args.is_gp:
        print("Parsing GP format...")
        if 'cache' not in data:
            raise ValueError("The JSON file does not contain a 'cache' key. Is this really a GP output?")
        
        # cache is a dict of { "expression_string": score }
        cache = data['cache']
        
        # Sort the cache by score (value) in descending order
        # Assuming higher score is better. If some are NaN or None, filter them out first.
        valid_cache = {k: v for k, v in cache.items() if v is not None and not (isinstance(v, float) and v != v)} # filter out nan
        
        sorted_alphas = sorted(valid_cache.items(), key=lambda item: item[1], reverse=True)
        
        # Take the top N
        top_alphas = sorted_alphas[:args.top_n]
        extracted_exprs = [alpha[0] for alpha in top_alphas]
        
        print(f"Extracted {len(extracted_exprs)} alphas from GP cache (highest score: {top_alphas[0][1]:.4f} if available).")
        
    else:
        print("Parsing standard (PPO/GFN) format...")
        # PPO and GFN usually save a dict with 'exprs'
        if 'exprs' in data:
            exprs = data['exprs']
            # We assume they are already sorted by the pool, or we just take the top N available
            extracted_exprs = exprs[:args.top_n]
            print(f"Extracted {len(extracted_exprs)} alphas from standard format.")
        else:
            raise ValueError("The JSON file does not contain an 'exprs' key. Check the format.")

    # Create the standard output format required by evaluate_trophies.py
    out_data = {
        'exprs': extracted_exprs,
        'weights': [1.0] * len(extracted_exprs)
    }

    # Determine output path
    out_path = args.out_file
    if out_path is None:
        base, ext = os.path.splitext(args.file)
        out_path = f"{base}_top{args.top_n}{ext}"

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=4)
        
    print(f"Successfully saved {len(extracted_exprs)} alphas to {out_path}")
    print(f"You can now run: python evaluate_trophies.py --expressions_file {out_path} --instruments csi300")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract top N alphas from GP, PPO, or GFN output JSONs.")
    parser.add_argument('--file', type=str, required=True, help="Path to the JSON file to parse")
    parser.add_argument('--is_gp', action='store_true', help="Flag to indicate if the file is from train_GP.py")
    parser.add_argument('--top_n', type=int, default=50, help="Number of top alphas to extract (default: 50)")
    parser.add_argument('--out_file', type=str, default=None, help="Output JSON path (optional)")
    
    args = parser.parse_args()
    extract_top_alphas(args)
