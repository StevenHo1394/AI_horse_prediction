# AI_horse_prediction

 AI Horse racing model (using 3-layer Neural Network, predict the finishing position of each horse in a race)

  -package: 
          ./saved_model/*: pre-trained model using 20,000 epochs;
          horse_racing_model4.py: The NN model for horse racing;
          evaluate_model.py: To evaluate the performance of the model; 
          predict3.py: Predict the results

  -usage: 
          1) make sure the pre-trained model is present. If not, train the model by running "horse_racing_model4.py";
          2) (optional) evaluate the model by running "evaluate_model.py"
          3) (if new prediction needed) add a new csv, name as "horse_data_yyyymmdd_racex.csv" with the same format as the example "horse_data_20201202_race2.csv";
          modify the line "dataset = pd.read_csv('horse_data_20201202_race2.csv')" to "horse_data_yyyymmdd_racex.csv";
          4) run the "predict3.py" and inspect the predicted results on console;
          
  -more:
          This is an ongoing project and will be updated from time to time.

Accuracy:
=================
Model: v1.1
Data: v1.0.1
Date: 2021-02-05

Training: 38.63%
Testing: 28.81%

=================


Model: v1.0
Data: v1.0.0
Date: 2020-12-02

Training: 100%
Testing: 47%


  -improvements: need grab data from jockey club automatically, improve prediction accuracy, regularization, tackle "same finsihing position" of different horses...
