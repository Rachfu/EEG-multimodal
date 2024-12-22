from base_train import TrainAndTest

"""
compare:
    multimodal_type: "ti", "tt", "it", "ii"
fix: 
    txt_model: bert, txt_model_coef:"bert-base-uncased", 
    img_model: clip, img_model_coef:"ViT-B/32", 
    cross_atn_type: double_stream
    DP scheme: feature-level element-wise DP dropout
    privacy budget: epsilon = 0.1
"""

class CompareModal(object):
    def __init__(self,
                 train_type = "compare_modal",
                 txt_model="bert",
                 txt_model_coef="bert-base-uncased",
                 img_model="clip",
                 img_model_coef="ViT-B/32",
                 cross_atn_type="double_stream",
                 dp_mode="dropout_laplacian",
                 epsilon=0.1):
        self.train_type = train_type
        self.txt_model = txt_model
        self.txt_model_coef = txt_model_coef
        self.img_model = img_model
        self.img_model_coef = img_model_coef
        self.cross_atn_type = cross_atn_type
        self.dp_mode = dp_mode
        self.epsilon = epsilon
        self.python_job = TrainAndTest()
    def test_ti(self):
        multimodal_type = "ti"
        eeg_model = self.txt_model
        eeg_model_coef = self.txt_model_coef
        act_model = self.img_model
        act_model_coef = self.img_model_coef
        self.python_job.train(self.train_type,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def test_tt(self):
        multimodal_type = "tt"
        eeg_model = self.txt_model
        eeg_model_coef = self.txt_model_coef
        act_model = self.txt_model
        act_model_coef = self.txt_model_coef
        self.python_job.train(self.train_type,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def test_it(self):
        multimodal_type = "it"
        eeg_model = self.img_model
        eeg_model_coef = self.img_model_coef
        act_model = self.txt_model
        act_model_coef = self.txt_model_coef
        self.python_job.train(self.train_type,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def test_ii(self):
        multimodal_type = "ii"
        eeg_model = self.img_model
        eeg_model_coef = self.img_model_coef
        act_model = self.img_model
        act_model_coef = self.img_model_coef
        self.python_job.train(self.train_type,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def run(self):
        self.test_ti()
        self.test_tt()
        self.test_it()
        self.test_ii()

if __name__ == "__main__":
    print("I'm running to compare modal choices within ti,tt,it,ii...")
    python_job = CompareModal()
    python_job.run()
