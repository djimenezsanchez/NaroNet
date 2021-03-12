import matplotlib.pyplot as plt
import numpy as np

def plot_roc(fpr,tpr,roc_auc):
    plt.figure()
    lw = 3
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.title('Receiver operating characteristic',fontsize=16)
    plt.legend(loc="lower right",fontsize=16)    
    return plt

def plot_prec_recall(recall,precision,AP,f1_score):
    plt.figure()
    lw = 3
    plt.plot(recall, precision, color='navy',
            lw=lw, label='P-R curve (AP = %0.2f)' % AP)    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall',fontsize=16)
    plt.ylabel('Precision',fontsize=16)
    plt.title('Precision-Recall curve (F1-score: {:.2f})'.format(f1_score),fontsize=16)
    plt.legend(loc="lower left",fontsize=16)    
    return plt

def plot_training_loss_acc(train_info=[],val_info=[], title=[], label=[], ylabel=[]):    
    train_info_mean = [np.mean(t) for t in train_info]
    train_info_std = [np.std(t) for t in train_info]
    val_info_mean = [np.mean(t) for t in val_info]
    val_info_std = [np.std(t) for t in val_info]
    plt.errorbar(np.linspace(1, len(train_info_mean), len(train_info_mean),endpoint=True),train_info_mean,yerr=train_info_std, label=label[0])
    plt.errorbar(np.linspace(1, len(val_info_mean), len(val_info_mean)),val_info_mean,yerr=val_info_std, label=label[1])
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(fontsize='large')
    return plt

