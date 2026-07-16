import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm

from src.alphagen_qlib.stock_data import StockData
from alphagen.data.expression import Feature, FeatureType, Ref
from src.gan.utils.builder import exprs2tensor
from run_adaptive_combination import load_alpha_pool_by_path

def compute_alpha_stats(alpha_tensor, returns_tensor):
    """
    Computes portfolio weights, daily PnL, turnover, and stock counts.
    alpha_tensor: (n_days, n_stocks)
    returns_tensor: (n_days, n_stocks)
    """
    # Cross-sectional demean
    alpha = alpha_tensor - alpha_tensor.nanmean(dim=1, keepdim=True)
    alpha = torch.nan_to_num(alpha, nan=0.0)
    
    # L1 normalization
    l1_norm = alpha.abs().sum(dim=1, keepdim=True)
    l1_norm[l1_norm == 0] = 1e-8
    weights = alpha / l1_norm
    
    # Daily PnL
    daily_returns = torch.nan_to_num(returns_tensor, nan=0.0)
    daily_pnl = (weights * daily_returns).sum(dim=1)
    
    # Turnover
    delta_weights = (weights[1:] - weights[:-1]).abs().sum(dim=1) / 2
    turnover = delta_weights.mean().item()
    
    # Stock Counts
    long_stocks = (weights > 1e-6).sum(dim=1).float().mean().item()
    short_stocks = (weights < -1e-6).sum(dim=1).float().mean().item()
    
    return daily_pnl, turnover, long_stocks, short_stocks, weights

def calculate_ir(daily_pnl):
    if len(daily_pnl) < 2:
        return 0.0
    mean_pnl = daily_pnl.mean().item()
    std_pnl = daily_pnl.std().item()
    if std_pnl == 0:
        return 0.0
    # Annualized IR (assuming ~252 trading days)
    # The prompt defines IR as mean(daily_pnl) / std(daily_pnl) 
    # Usually IR is annualized, but we will strictly follow the prompt definition: mean / std
    return mean_pnl / std_pnl

def run(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.instruments == 'sp500':
        QLIB_PATH = './data/qlib_data/us_data_qlib'
    else:
        QLIB_PATH = './data/qlib_data/cn_data_rolling'
        
    print(f"Loading data from {QLIB_PATH} for {args.instruments}...")
    
    # We load data starting early enough to capture the first period (2006-01-04)
    # Using 2005-01-01 to give padding for rolling features.
    data = StockData(
        instrument=args.instruments,
        start_time='2005-01-01',
        end_time='2017-12-31',
        qlib_path=QLIB_PATH,
        device=device
    )
    if data.max_future_days == 0:
        dates = data._dates[data.max_backtrack_days:]
    else:
        dates = data._dates[data.max_backtrack_days:-data.max_future_days]
    
    # Target definition: 1-day or 20-day returns based on user flag
    close = Feature(FeatureType.CLOSE)
    if args.target_days == 1:
        target = Ref(close, -1) / close - 1
    else:
        target = Ref(close, -args.target_days) / close - 1
        
    tgt_tensor = exprs2tensor([target], data, normalize=False)[..., 0].to(device)

    print(f"Loading expressions from {args.expressions_file}...")
    expressions, _ = load_alpha_pool_by_path(args.expressions_file)
    print(f"Loaded {len(expressions)} expressions.")
    
    fct_tensor = exprs2tensor(expressions, data, normalize=True).to(device)
    
    # Identify Period Masks
    dates_ts = pd.DatetimeIndex(dates)
    mask1 = (dates_ts >= '2006-01-04') & (dates_ts <= '2017-11-30')
    mask2 = (dates_ts >= '2011-01-03') & (dates_ts <= '2017-11-30')
    mask3 = (dates_ts >= '2013-04-01') & (dates_ts <= '2017-11-30')
    
    # Convert numpy mask to torch tensor for indexing
    mask1_t = torch.tensor(mask1, device=device)
    mask2_t = torch.tensor(mask2, device=device)
    mask3_t = torch.tensor(mask3, device=device)

    results = []
    daily_pnl_all = []
    
    print("Evaluating Trophy Constraints...")
    for i in tqdm(range(fct_tensor.shape[-1])):
        alpha_slice = fct_tensor[..., i]
        
        daily_pnl, turnover, long_stocks, short_stocks, weights = compute_alpha_stats(alpha_slice, tgt_tensor)
        daily_pnl_all.append(daily_pnl)
        
        # Calculate IR for the three periods
        pnl1 = daily_pnl[mask1_t]
        pnl2 = daily_pnl[mask2_t]
        pnl3 = daily_pnl[mask3_t]
        
        ir1 = calculate_ir(pnl1)
        ir2 = calculate_ir(pnl2)
        ir3 = calculate_ir(pnl3)
        
        total_stocks = long_stocks + short_stocks
        
        # Constraints check
        pass_ir = (ir1 > 0.07) and (ir2 > 0.07) and (ir3 > 0.07)
        pass_to = (turnover < 0.50)
        pass_stocks = (total_stocks > 200)
        
        is_trophy = pass_ir and pass_to and pass_stocks
        
        results.append({
            'idx': i,
            'expr': str(expressions[i]),
            'ir1': ir1, 'ir2': ir2, 'ir3': ir3,
            'turnover': turnover,
            'long_stocks': long_stocks,
            'short_stocks': short_stocks,
            'total_stocks': total_stocks,
            'is_trophy': is_trophy
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    trophy_df = df[df['is_trophy']].copy()
    
    print("\n--- Initial Trophy Evaluation ---")
    print(f"Total Alphas: {len(df)}")
    print(f"Passed Initial Constraints (IR, TO, Stocks): {len(trophy_df)}")
    
    if len(trophy_df) == 0:
        print("No Trophy Alphas found. Loosen constraints to see borderline alphas.")
        # Print top 5 borderline alphas by average IR just for reference
        df['avg_ir'] = (df['ir1'] + df['ir2'] + df['ir3']) / 3
        print("\nTop 5 Closest Alphas (by Avg IR):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.sort_values(by='avg_ir', ascending=False).head(5)[['idx', 'ir1', 'ir2', 'ir3', 'turnover', 'total_stocks']])
        return

    # Evaluate Mutual Correlations (< 50%)
    print("\nEvaluating Mutual Correlations...")
    # Stack the daily PnLs of the trophy alphas to calculate correlation
    trophy_indices = trophy_df['idx'].values
    trophy_pnls = torch.stack([daily_pnl_all[idx] for idx in trophy_indices], dim=1) # (n_days, n_trophies)
    
    # Calculate correlation matrix using torch.corrcoef (requires shape [n_vars, n_obs])
    corr_matrix = torch.corrcoef(trophy_pnls.T).cpu().numpy()
    
    # Greedily select alphas to ensure < 50% correlation
    # We will prioritize alphas with the highest average IR across all 3 periods
    trophy_df['avg_ir'] = (trophy_df['ir1'] + trophy_df['ir2'] + trophy_df['ir3']) / 3
    sorted_trophy_df = trophy_df.sort_values(by='avg_ir', ascending=False)
    
    final_selected_idx = []
    final_selected_original_idx = []
    
    for i, row in sorted_trophy_df.iterrows():
        orig_idx = row['idx']
        matrix_idx = np.where(trophy_indices == orig_idx)[0][0]
        
        # Check correlation with already selected alphas
        conflict = False
        for sel_orig_idx in final_selected_original_idx:
            sel_matrix_idx = np.where(trophy_indices == sel_orig_idx)[0][0]
            corr = corr_matrix[matrix_idx, sel_matrix_idx]
            if corr >= 0.50:
                conflict = True
                break
                
        if not conflict:
            final_selected_idx.append(i)
            final_selected_original_idx.append(orig_idx)

    final_df = sorted_trophy_df.loc[final_selected_idx]
    
    print("\n--- Final Trophy Alphas (Post-Correlation Filter) ---")
    print(f"Final Count: {len(final_df)}")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(final_df[['idx', 'expr', 'ir1', 'ir2', 'ir3', 'turnover', 'total_stocks']])

    # Save to JSON
    out_path = args.expressions_file.replace('.json', '_trophy.json')
    with open(out_path, 'w') as f:
        output_data = {
            'exprs': final_df['expr'].tolist(),
            'weights': [1.0] * len(final_df)
        }
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved Final Trophy Alphas to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expressions_file', type=str, required=True, help="Path to the JSON pool of alphas")
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--target_days', type=int, default=1, help="Holding period for target returns (1 for daily PnL, 20 for original training target)")
    
    args = parser.parse_args()
    run(args)
