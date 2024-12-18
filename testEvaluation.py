import torch
import os
import time
from train import *
import segmentation_models_pytorch as smp
torch.backends.cudnn.benchmark = False
from utils.train_utlis import *
from utils.data_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import jaccard_score , confusion_matrix
from utils.plot_utils import *



#test data
def test_evaluation(train_dir,cfg):
        #config
        data_dir = cfg['data']['dir']
        desirable_class = cfg['train']['desirable_class']
        #make dir for the plots
        metric_path = os.path.join(train_dir,'metrics')
        os.makedirs(metric_path, exist_ok=True)

        #labels
        labels = [i for i in range(desirable_class)]
        
       #load checkpoints models
        model = load_model(cfg)
        models_list = [os.path.join(train_dir, item) for item in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, item))]
        models_dict = {item: [] for item in models_list}        #load test data
        _, test_loader = load_data(cfg,desirable_class,1,data_dir,test_mode = True)

       
        for model_path in models_list:
                #predict the test data
                metrics = []
                prdeictions = []
                targets = []
                acc_timer = 0
                model.load_state_dict(torch.load(model_path))
                model.to(device)
                model.eval()
                for input, target in test_loader:
                        input = input.to(device)
                        target = (target.squeeze(1)).cpu().numpy()
                        target = target.flatten()
                        with torch.no_grad():
                                #predict and take time
                                start_time = time.time()
                                output = predict(model, input)
                                end_time = time.time() - start_time
                                acc_timer += end_time
                                output = output.cpu().numpy()
                                output = output.flatten()
                                prdeictions.append(output)
                                targets.append(target)
                

                #concatenate the predictions and targets
                prdeictions = np.concatenate(prdeictions)
                targets = np.concatenate(targets)

                #confusion matrix
                model_confusion_matrix = confusion_matrix(targets,prdeictions,labels=labels)
                metrics.append(model_confusion_matrix)
                
                #pixel accuracy
                true_positive = np.trace(model_confusion_matrix)  # Sum of diagonal elements (True positives)
                total_pixels = np.sum(model_confusion_matrix)     # Sum of all elements in the matrix
                pixel_accuracy = true_positive / total_pixels
                metrics.append(pixel_accuracy)

                #jacard (iou)
                
                iou_micro = jaccard_score(targets, prdeictions, average='micro', labels=labels)
                iou_weighted = jaccard_score(targets, prdeictions, average='weighted', labels=labels)
                iou_per_class = []
                for label in labels:
                        jaccard_class = jaccard_score(targets, prdeictions, average=None, labels=[label])[0]
                        iou_per_class.append(jaccard_class)
                lowest_iou = np.min(iou_per_class)
                metrics.append(iou_micro)
                metrics.append(iou_weighted)
                metrics.append(lowest_iou)

                #recall , precision, f1 score (dice)
                recall = np.mean(true_positive / np.sum(model_confusion_matrix, axis=1))
                precision = np.mean(true_positive / np.sum(model_confusion_matrix, axis=0))
                f1_score = 2 * (precision * recall) / (precision + recall)
                metrics.append(recall)
                metrics.append(precision)
                metrics.append(f1_score)

                #average time
                avg_time = acc_timer / len(test_loader)
                metrics.append(avg_time)

                #plotting and save
                plot_confusion_matrix_with_metrics(metrics[:-1], model_path,metric_path)
                models_dict[model_path] = metrics
        
        #compare between models
        compare_models_performers(models_dict, metric_path)
        return




if __name__ == "__main__":
        train_dir = '/workspace/results'
        model = load_model(cfg) 
        test_evaluation(train_dir,model,cfg)






        


