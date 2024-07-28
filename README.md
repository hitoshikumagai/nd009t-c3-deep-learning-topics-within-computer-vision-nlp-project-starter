# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
### About Model in this project 
I selected RESNET18 because it has simple and efficient structure
Number of Layers: Despite the relatively shallow number of layers (18), it performs as well as a deep network. This is achieved through a structure called residual blocks.
### Tuned Hyperparameter
* learning rate: [0.01, 0.1]
    * Definition: The step size for updating the parameters of the model.
    * Role: If the learning rate is too large, oscillations will occur and convergence to an optimal solution will be difficult. Conversely, if it is too small, learning will be very slow.
* momentum: [0.01, 0.9]
    * Definition: A technique to add inertia in the direction of parameter updates to stabilize the direction to the optimal solution and prevent falling into a local solution.
    * Role: Introducing momentum allows for smoother movement to the optimal solution in error functions where the parameters are trough-like in shape.

### Result
![completed training jobs](image/hpo_loss.png)
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
