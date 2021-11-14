




# GAN, ACGAN and UDA
In this assignment, you are given datasets of human face and digit images. You will need to implement the models of both GAN and ACGAN for generating human face images, and the model of DANN for classifying digit images from different domains.

<p align="center">
  <img width="550" height="500" src="p1p2/fig1_2.jpg">
</p>

For more details, please click [this link](https://1drv.ms/p/s!AmVnxPwdjNF2gZtOUMO5HEEQqLB8Ew) to view the slides of HW3.

# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

    git clone https://github.com/dlcv-spring-2019/hw3-<username>.git
Note that you should replace `<username>` with your own GitHub username.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw3_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/uc?export=download&id=1gbnGEMyLIsYdIoyUyVZjYK8MzQZs4e_V) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `hw3_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
To evaluate your UDA models in Problems 3 and 4, you can run the evaluation script provided in the starter code by using the following command.

    python3 hw3_eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `hw3_data/digits/mnistm/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw3_data/digits/mnistm/test.csv`)

Note that for `hw3_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.

| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

# Submission Rules
### Deadline
108/05/08 (Wed.) 01:00 AM

### Late Submission Policy
You have a five-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade. For more information, please check out [this post](https://www.facebook.com/notes/dlcv-spring-2019/lateearly-homework-submission/326632628047121/).

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw3_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 1.   `hw3_p1p2.sh`  
The shell script file for running your GAN and ACGAN models. This script takes as input a folder and should output two images named `fig1_2.jpg` and `fig2_2.jpg` in the given folder.
 1.   `hw3_p3.sh`  
The shell script file for running your DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.
 1.   `hw3_p4.sh`  
The shell script file for running your improved UDA model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.

We will run your code in the following manner:

    bash ./hw3_p1p2.sh $1
    bash ./hw3_p3.sh $2 $3 $4
    bash ./hw3_p4.sh $2 $3 $4

-   `$1` is the folder to which you should output your `fig1_2.jpg` and `fig2_2.jpg`.
-   `$2` is the directory of testing images in the **target** domain (e.g. `hw3_data/digits/mnistm/test`).
-   `$3` is a string that indicates the name of the target domain, which will be either `mnistm`, `usps` or `svhn`. 
	- Note that you should run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `mnistm`, you should make your prediction using your "USPS→MNIST-M" model, **NOT** your "MNIST-M→SVHN" model.
-   `$4` is the path to your output prediction file (e.g. `hw3_data/digits/mnistm/test_pred.csv`).


