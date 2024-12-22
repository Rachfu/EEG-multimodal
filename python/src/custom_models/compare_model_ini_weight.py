import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from base_train import TrainAndTest

"""
compare:
    models with different initial weights:
        bert: "bert-base-uncased", "bert-base-cased"
        vit: "ViT-B/32", "ViT-B/16"
        resnet: resnet34
fix:
    multimodal_type: "ti"
    cross_atn_type: double_stream
    DP scheme: feature-level element-wise DP dropout
    privacy budget: epsilon = 0.1
"""
class CompareModelIniWeight(object):
    def __init__(self,
                 train_type = "compare_model_ini_wight",
                 multimodal_type = "ti",
                 cross_atn_type="double_stream",
                 dp_mode="dropout_laplacian",
                 epsilon=0.1):
        self.train_type = train_type
        self.multimodal_type = multimodal_type
        self.cross_atn_type = cross_atn_type
        self.dp_mode = dp_mode
        self.epsilon = epsilon
        self.python_job = TrainAndTest()
    def run(self):
        for txt_model,txt_coef in [["bert","bert-base-uncased"],["bert","bert-base-cased"]]:
            for img_model,img_coef in [["clip","ViT-B/32"],["clip","ViT-B/16"],["resnet","resnet34"]]:
                print(txt_model,txt_coef,img_model,img_coef)
                self.python_job.train(self.train_type,self.multimodal_type,self.dp_mode,txt_model,txt_coef,img_model,img_coef,self.cross_atn_type,self.epsilon)

if __name__ == "__main__":
    print("I'm running to compare model coef choices within diff initial weights of bert, clip, and plus test for resnet")
    python_job = CompareModelIniWeight()
    python_job.run()