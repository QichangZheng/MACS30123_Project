# Q1 ([Code](https://github.com/macs30123-s23/assignment-3-QichangZheng/blob/main/Q1.ipynb))

# Q234 ([Code](https://github.com/macs30123-s23/assignment-3-QichangZheng/blob/main/Q234.ipynb))

# Q3(a)
### Why I include the features:
#### total_votes: This feature represents the total number of votes a review has received. It can be relevant because reviews with a higher number of votes may indicate a higher level of engagement or popularity. This feature captures the overall attention or interest the review has garnered from readers.
#### title_len: This feature represents the length of the product title associated with each review. The length of the title can provide additional context and information about the product being reviewed. It is possible that the length of the title may influence the perception or impact of the review on readers.
#### helpful_prop: This feature calculates the proportion of helpful votes out of the total votes received for each review. It captures the extent to which other readers found the review helpful. A higher proportion of helpful votes suggests that the review is more informative or valuable to readers, potentially indicating a good review.
#### verified_pur: This feature indicates whether the review is from a verified purchase. Reviews from verified purchases are typically given more credibility and trustworthiness as they come from customers who have actually purchased the product. This feature helps capture the influence of verified purchases on determining good reviews.

# Q3(c)

#### In Spark, when we specify a series of transformations in a pipeline, the DataFrame is not processed immediately. This is because Spark operates on the principles of lazy evaluation. This means it only computes the results when an action is called (like count, collect, show, or fit in the provided code), not when transformations (like withColumn, sampleBy, or transform) are defined.
#### The operations we define are recorded in a Directed Acyclic Graph (DAG). When an action is triggered, Spark optimizes this DAG to determine the most efficient way to execute the operations, considering factors such as data locality and partitioning. This is part of what allows Spark to handle large-scale data processing efficiently.
#### In the given Spark code, the transformations are not actually executed until show is called on the DataFrame sampled and fit is called on the model in the Train function. persist is used to cache the intermediate data after each transformation, which can speed up computation especially when the data is reused multiple times in subsequent transformations or actions.
#### Comparatively, Dask also operates on principles of lazy evaluation and uses a similar task scheduling approach. However, Dask's execution model is more dynamic and flexible than Spark's, which is more rigid and based on the two-stage MapReduce paradigm. Dask builds a dynamic task graph that can adapt to handle complex and irregular computation patterns that don't fit neatly into a MapReduce model.
#### In the Dask structure, computations are not performed until compute is called, similar to an action in Spark. Once compute is called, Dask constructs the task graph, optimizes it, and schedules the tasks for execution, returning a Pandas DataFrame with the results.
#### In summary, both Spark and Dask follow a lazy evaluation model, building a task graph for computation and only executing when an action is triggered. The key difference lies in their scheduling and execution models, with Dask being more dynamic and adaptable to irregular computations, while Spark adheres more strictly to the MapReduce model.

# Q4
### For Label 0:
#### False Positive Rate: 47.14%. This means that out of all instances that were actually not label 0, the model incorrectly predicted them as label 0 around 47.14% of the time.
#### True Positive Rate (also known as Sensitivity or Recall): 76.64%. This means that out of all instances that were actually label 0, the model correctly predicted them as label 0 about 76.64% of the time.
### For Label 1:
#### False Positive Rate: 23.36%. This means that out of all instances that were actually not label 1, the model incorrectly predicted them as label 1 about 23.36% of the time.
#### True Positive Rate: 52.86%. This means that out of all instances that were actually label 1, the model correctly predicted them as label 1 about 52.86% of the time.
### What the model does well:
#### It has a relatively high true positive rate for Label 0, meaning it does a decent job at correctly identifying instances of this class.
### What the model does poorly:
#### The model has a high false positive rate for Label 0, meaning it often misclassifies instances as belonging to this class when they do not.
#### The model has a relatively low true positive rate for Label 1, suggesting that it often misses instances of this class.
### Ways to improve the model:
#### To improve the low true positive rate for Label 1, we could explore feature engineering to create more relevant predictors, or try different modeling techniques that may be better suited to our specific problem. We could also tune the model hyperparameters to see if that improves performance. Lastly, it would be worth reviewing the loss function and evaluation metrics to ensure they are correctly incentivizing the model's predictions. For example, we may want to penalize false positives more heavily if they are particularly costly in our application.


