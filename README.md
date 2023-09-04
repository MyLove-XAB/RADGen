# RADGen
**RADGen(REINFORCE Aided Deep Generator for news headline)**  
Aobo Xu, Ling Jian. A Deep News Headline Generation Model with REINFORCE Filter. 2023 International Joint Conference on Neural Networks (IJCNN).IEEE, 2023. (DOI: 10.1109/IJCNN54540.2023.10192007)

## Package Requirement
tensorflow==2.4.1  
wandb==0.12.2  
rouge==1.0.1  
numpy==1.19.5  
pandas=1.1.1  

## Code Running Order

**1. tokenization.py**

**2. LoadDataChinese.py**

**3. TransformerChinese.py**: Training Stage One

**4. baseline_val.py, baseline_test.py, baseline_train.py**

**5. TrainRLChinese_R1.py, TrainRLChinese_R1_sample.py**: Training Stage Two

## Cite
**BibTex**  
@inproceedings{xu2023deep,
  title={A Deep News Headline Generation Model with REINFORCE Filter},
  author={Xu, Aobo and Jian, Ling},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--7},
  year={2023},
  organization={IEEE}
}