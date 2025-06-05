# RLsite: RNA language model and Graph attention network for RNA and small molecule binding sites prediction


# Requirements

conda env create -f environment.yml

# Usage

# Making your prediction using the pre-trained model 

$ unzip data/training&test_data.zip

$ python run.py --load_path final_model.pth --test_path data/test.txt

This will output the prediction results of ids in test.txt.

# Citation
Saisai Sun, Jianyi Yang, Lin Gao, et al., RNA language model and Graph attention network for RNA and small molecule binding sites prediction, In submission.
