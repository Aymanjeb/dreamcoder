# Applying Dreamcoder to ARC Challenge

It's clear that Dreamcoder's successful execution relies on multiple files and software dependencies, as documented in the "docs" folder. To properly run Dreamcoder, following the instructions outlined in the original ReadMe file on the master branch is crucial. While attempting to use a singularity container, I encountered challenges with the provided recipe file. This issue led me to make adjustments to certain library versions, as detailed in this GitHub issue: https://github.com/ellisk42/ec/issues/96.

In terms of modifications, I've taken steps to enhance the architecture of the Neural Network within the "dreamcoder/domains/arc/main.py" file. Additionally, during the training phase, I've incorporated augmented data. To showcase the new neural network architecture without data augmentation, I've created a file named "main_without_augmentation."

For the purpose of recording outcomes, I've introduced specific commands to store the results I've obtained in the "all_results" folder. The subfolders within "all_results" serve distinct purposes:

    "NN_results": This directory contains the predictions made by the Neural Network following the training process, displaying the probabilities associated with each primitive.
    "recognition_enumeration": Within this folder, you'll find .txt files that indicate whether each task was resolved during the enumeration process after training the Neural Network.
    "enumeration_results": Here, you'll come across .txt files that signify whether each task was solved through the enumeration process after updating the scores of primitives.
    
