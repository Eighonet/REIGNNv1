from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score, accuracy_score

def train(model, optimizer, criterion, batch_list_x, batch_list_owner, mt_coeffs, main_coeff, operator):
    """
    Perform one train epoch.
      *  model     -- instance of REIGNN class. 
      *  optimizer -- instance of optimizer.
      *  criterion -- loss function.
      *  mt_coeffs -- weights of auxiliry task losses.
      *  main_coeff -- weight of main task loss.
    """
    model.train()
    optimizer.zero_grad()
    sample = model.train_data_a
    z, z_sjr, z_hi, z_ifact, z_numb = model(sample, batch_list_x, batch_list_owner, operator)
    edge_index = sample.edge_label_index
    link_labels = sample.edge_label
    main_loss = main_coeff*F.binary_cross_entropy_with_logits(z, link_labels)
    mask = z > 0
    loss = main_loss\
           + mt_coeffs[0]*criterion(z_sjr[mask], sample.aux[1][mask])\
           + mt_coeffs[1]*criterion(z_hi[mask], sample.aux[2][mask])\
           + mt_coeffs[2]*criterion(z_ifact[mask], sample.aux[3][mask])\
           + mt_coeffs[3]*criterion(z_ifact[mask], sample.aux[4][mask])
    loss.backward()
    optimizer.step()
    return loss
    
@torch.no_grad()
def test(model, optimizer, criterion, batch_list_x, batch_list_owner, operator):
    """
    Perform evaluation.
      *  model     -- instance of REIGNN class. 
      *  optimizer -- instance of optimizer.
      *  criterion -- loss function.
    """
    model.eval()
    perfs = []
    aux = []

    for sample in [model.train_data_a, model.test_data_a]: 
        z, z_sjr, z_hi, z_ifact, z_numb = model(sample, batch_list_x, batch_list_owner, operator)
        link_probs = z.sigmoid()
        link_labels = sample.edge_label
        aux.append([mean_absolute_error(sample.aux[1].cpu(), z_sjr.cpu()),\
                    mean_absolute_error(sample.aux[2].cpu(), z_hi.cpu()),\
                    mean_absolute_error(sample.aux[3].cpu(), z_ifact.cpu()),\
                    mean_absolute_error(sample.aux[4].cpu(), z_numb.cpu())])
        perfs.append([accuracy_score(link_labels.cpu(), link_probs.cpu().round()),\
                      f1_score(link_labels.cpu(), link_probs.cpu().round()),\
                      roc_auc_score(link_labels.cpu(), link_probs.cpu())])
    return perfs, aux

def run(wandb_output:bool, project_name:str, group:str, entity:str, mt_weights:list, model, optimizer, criterion, operator, batch_list_x, batch_list_owner, N:int) -> None:
    """
    Launch train process for N epochs.
      *  wandb_output -- wandb output flag.
      *  project_name -- name of wandb project.
      *  group        -- group in wandb project.
      *  entity       -- entity in wandb project.
      *  mt_weights   -- weights of auxiliry task losses.
      *  model        -- instance of REIGNN class. 
      *  optimizer    -- instance of optimizer.
      *  criterion    -- loss function.
      *  N            -- number of train epochs.
    """
    if wandb_output:
        wandb.init(project=project_name, entity=entity, group=group)
        wandb.run.name = group + "_" + str(i)
        wandb.run.save()
    
    max_acc_test, max_f1_test, max_roc_auc_test = 0, 0, 0
    max_acc_val, max_f1_val, max_roc_auc_val = 0, 0, 0

    min_mae_sjr_test, min_mae_h_index_test, min_mae_impact_factor_test, min_mae_number_test = 100500, 100500, 100500, 100500
    min_mae_sjr_val, min_mae_h_index_val, min_mae_impact_factor_val, min_mae_number_val = 100500, 100500, 100500, 100500
    
    for epoch in tqdm(range(N)):
        loss = []
        train_loss = train(model, optimizer, criterion, batch_list_x, batch_list_owner, mt_weights, 1, operator)

        if epoch % 10 == 0:
            metrics, metrics_aux = test(model, optimizer, criterion, batch_list_x, batch_list_owner, operator)
            print("Train main:", metrics[0])
            print("Test main:", metrics[1])
            print("Train auxiliary:", metrics_aux[0])
            print("Test auxiliary:", metrics_aux[1])
        
        if metrics[1][0] > max_acc_test:
            max_acc_test = metrics[1][0]
        if metrics[1][1] > max_f1_test:
            max_f1_test = metrics[1][1]
        if metrics[1][2] > max_roc_auc_test:
            max_roc_auc_test = metrics[1][2]  
        
        if metrics_aux[1][0] < min_mae_sjr_test:
            min_mae_sjr_test = metrics_aux[1][0]
        if metrics_aux[1][1] < min_mae_h_index_test:
            min_mae_h_index_test = metrics_aux[1][1]
        if metrics_aux[1][2] < min_mae_impact_factor_test:
            min_mae_impact_factor_test = metrics_aux[1][2]  
        if metrics_aux[1][3] < min_mae_number_test:
            min_mae_number_test = metrics_aux[1][3]
            
        if wandb_output:
            wandb.log({"main_train/train_acc":  metrics[0][0], "main_test/test_acc": metrics[1][0],\
                   "main_train/train_f1": metrics[0][1], "main_test/test_f1": metrics[1][1],\
                   "main_train/train_roc_auc": metrics[0][2], "main_test/test_roc_auc": metrics[1][2]})
            
            wandb.log({"main_max/test_max_acc": max_acc_test,\
                   "main_max/test_max_f1": max_f1_test,\
                   "main_max/test_max_roc_auc": max_roc_auc_test})
        
            wandb.log({"aux_train/train_mae_sjr":  metrics_aux[0][0], "aux_train/train_mae_h_index": metrics_aux[0][1], "aux_train/train_mae_impact_factor": metrics_aux[0][2], "aux_train/train_number": metrics_aux[0][3],
                   "aux_test/test_mae_sjr":  metrics_aux[1][0], "aux_test/test_mae_h_index": metrics_aux[1][1], "aux_test/test_mae_impact_factor": metrics_aux[1][2], "aux_test/test_number": metrics_aux[1][3]})
            
            wandb.log({"aux_min/test_min_mae_sjr": min_mae_sjr_test,\
                   "aux_min/test_min_mae_h_index": min_mae_h_index_test,\
                   "aux_min/test_min_mae_impact_factor": min_mae_impact_factor_test,
                   "aux_min/test_min_number_factor": min_mae_number_test})