
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="Instacart">
    <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3333/media/instacart_sidebar.png" alt="Logo" width="80" height="80">
  </a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
      <li>
      <a href="#Structure">Code folder Structure</a>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## About The Project

[Instacart market basket analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)

The objective is to predict the product that the user is most likely to order in their future order
based on their prior buying pattern. The dataset that has been shared to us contains orders of
200000 Instacart users with each user having between 4 and 100 order.Each user order is
indicated by prior, test, train where prior means order that was made previously train and test
are future data that we can train and test models. Here our target variable is reordered it tell us
whether the user reordered the product or not ,reordered is binary value 1 indicating that user
reordered the product and 0 indicates that user did not reorder the product,in some order
reorder value is missing meaning we do not have any information about reorder.
We will be training a binary classification model to predict if the user reordered the product or
not.For model selection we will select the model that gives the best  F1 score.

# Leaderboard score 0.355280


## Structure

instacart_fe_tuning.ipynb : Feature engineering and model tunning and training file <br>
eda_files - folder : Contains Instacart Eda<br>
final_model_for_deployment: folder contains prototype code for for making model ready for deployment using best model.<br>
model_deployment_code : folder contains code and configuration to deploy code in Aws EC2  <br>

<!-- CONTACT -->
## Contact

Aditya Kumar Ghosh - [@Linekdin](https://www.linkedin.com/in/aditya-ghosh-955b5b161/) - adityaghoshsbg@gmail.com
