import torch
import os
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from define_datasetclass import SegmentationDataset
from train import *
import segmentation_models_pytorch as smp
torch.backends.cudnn.benchmark = False
from utils import *
from pred_utiliz import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import jaccard_score , confusion_matrix
from plot_utiliz import *



#test data
def test_evaluation(train_dir,model,cfg):
        #config
        data_dir = cfg['data']['dir']
        desirable_class = cfg['train']['desirable_class']
        example_list = cfg['test_evaluation']['example_list']
        
       #load checkpoints models
        models_list = [os.path.join(train_dir,item) for item in os.listdir(train_dir)]
        models_dict = {item: [] for item in models_list}        #load test data
        _, test_loader = load_data(cfg,1,desirable_class,data_dir)

        for model_path in models_list:
                #predict the test data
                prdeictions = []
                targets = []
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()
                for (input, target) in test_loader:
                        input = input.to(device)
                        target = target.to(device)
                        with torch.no_grad():
                                output = predict(model,input)
                                prdeictions.append(output)
                                targets.append(target)

                #confusion matrix
                model_confusion_matrix = confusion_matrix(targets,prdeictions)

                #pixel accuracy
                true_positive = np.trace(model_confusion_matrix)  # Sum of diagonal elements (True positives)
                total_pixels = np.sum(model_confusion_matrix)     # Sum of all elements in the matrix
                pixel_accuracy = true_positive / total_pixels

                #jacard (iou)
                labels = [i for i in range(desirable_class)]
                iou_micro = jaccard_score(targets, prdeictions, average='micro', labels=labels)
                iou_weighted = jaccard_score(targets, prdeictions, average='weighted', labels=labels)
                iou_per_class = []
                for label in labels:
                        jaccard_class = jaccard_score(targets, prdeictions, average=None, labels=[label])[0]
                        iou_per_class.append(jaccard_class)
                lowest_iou = np.min(iou_per_class)
                        
                #recall , precision, f1 score (dice)
                recall = true_positive / np.sum(model_confusion_matrix, axis=1)
                precision = true_positive / np.sum(model_confusion_matrix, axis=0)
                f1_score = 2 * (precision * recall) / (precision + recall)

                #give exmples of the predictions
                
                
                #plotting and save
                plot_confusion_matrix_with_metrics(model_confusion_matrix, pixel_accuracy, iou_micro, iou_weighted, lowest_iou, recall, precision, f1_score, model_path)
                models_dict[model_path] = [pixel_accuracy,iou_micro,iou_weighted,lowest_iou,recall,precision,f1_score]
        
        #compare between models
        compare_models_performers(models_dict, train_dir)


        


