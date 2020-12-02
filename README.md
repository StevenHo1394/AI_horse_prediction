# AI_horse_prediction

 AI Horse racing model (using 3-layer Neural Network, predict the finishing position of each horse in a race)

  -package: 
          ./saved_model/*: pre-trained model using 20,000 epochs 
          horse_racing_model4.py: The NN model for horse racing
          evaluate_model.py: To evaluate the performance of the model 
          predict3.py: Predict the results

  -usage: 
          make sure the pre-trained model is present. If not, train the model by running "horse_racing_model4.py"
          add a new csv, name as "horse_data_yyyymmdd_racex.csv" with the same format as the example "horse_data_20201202_race2.csv"
          modify the line "dataset = pd.read_csv('horse_data_20201202_race2.csv')" to "horse_data_yyyymmdd_racex.csv"
          run the "predict3.py" and inspect the results on console 

  -improvements: grab data from jockey club automatically, improve accuracy, regularization...
