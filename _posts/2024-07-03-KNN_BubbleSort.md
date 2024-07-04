# KNN and Bubble Sort

A simple algorithm that can easily be leveraged for classification and regression, K nearest neighbors (KNN) seems to be as ubiqutious as logistic regression. KNN is known as a lazy algorithm, which merely means that there is not a training stage and the model is generated at prediction time. How can a model be generated when we want to make a prediction? KNN is non-parametric so there are no weights involved with the model, rather the training data itself is our model. To make this easier to understand we'll breakdown the steps to generate a KNN prediction given a new data point outside of our training dataset. 

As I've become more interested in C/C++ in the last year I'll leverage them for my implementation. We will only use a few standard libraries to create our own basic implementation from scratch.

# Define the Problem
We have a set of 6 homes that all have an id, square footage (sqft) and an associated price. 

```cpp
struct Home {
    int id = -1;
    int sqft = 0;
    int price = -1;
};

const int kNumberOfHomes = 6;
// Define the homes where square footage (feature) is used to predict price (label)
{% raw %}
std::array<Home, kNumberOfHomes> homes = {{
    {1, 35, 30000},
    {2, 45, 45000},
    {3, 40, 50000},
    {4, 35, 35000},
    {5, 25, 32500},
    {6, 40, 40000}
}};
{% endraw %}
```

We would like to be able to predict a home's price given the sqft of that home.


# Layout our Solution

Using KNN and a chosen similarity/distance metric we would like to be able to provide a price prediction. What are the steps for generating a prediction with KNN given our problem?

## Steps for prediction with KNN:
1. Find the k nearest neighbors according to a predefined similarity metric like euclidean distance.
    - Calculate distances between the point of interests features and all points in the dataset.
      ```cpp
      int CalculateEuclideanDistance(int x_1, int x_2) {
          return sqrt(pow(x_1 - x_2, 2));
      }

      // Calculate the 3 nearest neighbors for a new home outside of the training set using the home sq ft feature
      const int k = 3;
      int new_home_sqft = 32;
      std::array<Distance, kNumberOfHomes> neighbor_distances;
  
      for (int j=0; j<kNumberOfHomes; j++) {
          // calculate distance between the two homes according to features (just sqft)
          int distance = CalculateEuclideanDistance(new_home_sqft, homes[j].sqft);
          neighbor_distances[j] = {homes[j].id, distance};
          fmt::print("Home {}: (Sqft: {}, Price: {}) -> Distance: {}\n", j, homes[j].sqft, homes[j].price, neighbor_distances[j].distance);
      }
      ```
    - Sort the points according to the calculated distances.
      ```cpp
      void BubbleSort(Distance* distances, size_t size) {
          for (size_t i=0; i<size; ++i) {
              bool sorted = true;
              for (size_t j=0; j<size-i-1; ++j) {
                  if (distances[j].distance > distances[j+1].distance) {
                      Distance temp = distances[j];
                      distances[j] = distances[j+1];
                      distances[j+1] = temp;
                      sorted=false;
                  }
              }
              if (sorted) {
                  break;
              }
          }
      }
      
      BubbleSort(neighbor_distances.data(), neighbor_distances.size());
      std::array<int, kNumberOfHomes> sortedHomeIds;
      for (size_t i=0; i<neighbor_distances.size(); ++i) {
          sortedHomeIds[i] = neighbor_distances[i].homeId;
      }
      fmt::print("Homes sorted by distance: {}\n", sortedHomeIds);
      ```
    - Retain the top k points.
2. Apply a function to get your desired prediction (note our usecase is regression):
    - For regression avg, min, max or mean could be applied.
    - For classification we can just pick the most frequent class in the neighbor group (pick randomly if there is a tie).
      ```cpp
      int PredictHomePrice(int num_neighbors, Distance* distances, size_t size, Home* homes) {
          if (num_neighbors > size) {
              throw std::runtime_error("Size mismatch. More neighbors than homes available!\n");
          }
      
          int sum = 0;
          for (size_t i=0; i<num_neighbors; ++i) {
              Home home = getHomeById(distances[i].homeId, homes, size);
              sum += home.price;
          }
          return (sum / num_neighbors);
      }
      ```
  
The TLDR steps are simple. Calculate distances, sort based on the distances, keep top k points and apply a final function to obtain your prediction.

# Program Output for 32sqft Home

```
Home 0: (Sqft: 35, Price: 30000) -> Distance: 3
Home 1: (Sqft: 45, Price: 45000) -> Distance: 13
Home 2: (Sqft: 40, Price: 50000) -> Distance: 8
Home 3: (Sqft: 35, Price: 35000) -> Distance: 3
Home 4: (Sqft: 25, Price: 32500) -> Distance: 7
Home 5: (Sqft: 40, Price: 40000) -> Distance: 8
Homes sorted by distance: [1, 4, 5, 3, 6, 2]
For a home that is 32sqft the predicted price according to our dataset is 32500.
```

To review the final program and code you can go to the following file: [knn.cpp source link](https://github.com/jdspell/learningc/blob/main/machine_learning/KNN/knn.cpp).
