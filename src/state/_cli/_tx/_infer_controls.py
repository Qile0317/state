import argparse
from typing import Dict, List, Optional
import pandas as pd


def add_arguments_infer_controls(parser: argparse.ArgumentParser):
    parser.add_argument("--adata", type=str, required=True, help="Path to control cells AnnData file (.h5ad)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory containing checkpoint and mappings",
    )
    parser.add_argument(
        "--target-pert",
        type=str,
        required=True,
        help="Target perturbation to apply (e.g., 'Trametinib')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <input>_<pert>.h5ad)",
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Use adata.obsm[embed_key] as input (default: adata.X)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: model_dir/checkpoints/final.ckpt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )


def run_tx_infer_controls(args: argparse.Namespace):
    import os
    import pickle

    import numpy as np
    import scanpy as sc
    import torch
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    # -----------------------
    # Helpers
    # -----------------------
    def to_dense(mat):
        """Return a dense numpy array for a variety of AnnData .X backends."""
        try:
            import scipy.sparse as sp

            if sp.issparse(mat):
                return mat.toarray()
        except Exception:
            pass
        return np.asarray(mat)

    print(f"==> STATE: Apply '{args.target_pert}' to control cells")

    # Load model
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
    
    print(f"Loading model from {ckpt_path}")
    model = StateTransitionPerturbationModel.load_from_checkpoint(ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    
    # Load perturbation mapping
    pert_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    pert_onehot_map = torch.load(pert_map_path, weights_only=False)
    
    if args.target_pert not in pert_onehot_map:
        raise ValueError(f"Perturbation '{args.target_pert}' not in model vocabulary. Available: {list(pert_onehot_map.keys())[:10]}...")
    
    pert_vec = pert_onehot_map[args.target_pert].float()
    
    # Load data
    print(f"Loading control cells from {args.adata}")
    adata = sc.read_h5ad(args.adata)
    n_cells = adata.n_obs
    
    # Get input features
    if args.embed_key:
        if args.embed_key not in adata.obsm:
            raise KeyError(f"'{args.embed_key}' not in adata.obsm. Available: {list(adata.obsm.keys())}")
        X_in = np.asarray(adata.obsm[args.embed_key])
        print(f"Using adata.obsm['{args.embed_key}'] as input: {X_in.shape}")
    else:
        X_in = to_dense(adata.X)
        print(f"Using adata.X as input: {X_in.shape}")
    
    # Run inference
    print(f"Predicting {args.target_pert} on {n_cells} cells (batch_size={args.batch_size})...")
    predictions = []
    
    with torch.no_grad():
        for start in tqdm(range(0, n_cells, args.batch_size)):
            end = min(start + args.batch_size, n_cells)
            batch_size = end - start
            
            # Each cell uses its OWN expression
            ctrl_emb = torch.tensor(X_in[start:end], dtype=torch.float32, device=device)
            pert_emb = pert_vec.unsqueeze(0).repeat(batch_size, 1).to(device)
            
            batch = {
                "ctrl_cell_emb": ctrl_emb,
                "pert_emb": pert_emb,
                "pert_name": [args.target_pert] * batch_size,
            }
            
            output = model.predict_step(batch, batch_idx=0, padded=False)
            
            # Get predictions (prefer gene-space if available)
            if "pert_cell_counts_preds" in output and output["pert_cell_counts_preds"] is not None:
                pred = output["pert_cell_counts_preds"]
            else:
                pred = output["preds"]
            
            predictions.append(pred.cpu().numpy())
    
    predictions = np.vstack(predictions)
    print(f"Generated predictions: {predictions.shape}")
    
    # Store results
    if args.embed_key:
        adata.obsm[f"{args.embed_key}_pred"] = predictions
        print(f"Stored in adata.obsm['{args.embed_key}_pred']")
    else:
        if predictions.shape[1] == adata.n_vars:
            adata.layers[f"pred_{args.target_pert}"] = predictions
            print(f"Stored in adata.layers['pred_{args.target_pert}']")
        else:
            adata.obsm["X_state_pred"] = predictions
            print(f"Stored in adata.obsm['X_state_pred'] (dimension mismatch)")
    
    # Save
    if args.output:
        output_path = args.output
    else:
        safe_pert = args.target_pert.replace("/", "_").replace(" ", "_")
        output_path = args.adata.replace(".h5ad", f"_{safe_pert}.h5ad")
    
    adata.write_h5ad(output_path)
    print(f"\n✓ Saved to: {output_path}")
