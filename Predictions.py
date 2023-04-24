import pandas as pd
import os
import time
import math
import random

class Predictions:
    def __init__(self):
        self.test_ids_filepath = "data/test_ids.csv"
        self.train_labels_filepath = "data/train2023.csv"
        self.output_folder = "outputs/"
        self.model_path = "model.tf"
        # Choose value probabalistically, or by rounding
        # Options: 'round', 'prob'
        self.mode = 'round'
        
        self.ingest_test_csv()
        self.ingest_train_csv()
        self.load_model()


    def ingest_test_csv(self):
        df = pd.read_csv(self.test_ids_filepath, sep=',', header='infer')
        self.test_ids = df["Id"]
        self.paths_dict = dict(zip(df["Id"], df["Path"]))

    def ingest_train_csv(self):
        df = pd.read_csv(self.train_labels_filepath, sep=',', header='infer')
        self.train_df = df
        self.labels = df.columns[-9:]

    
    def load_model(self):
        # do something...
        self.model = 1


    def generate_predictions(self):
        # output_labels = ("Id") + self.labels
        # output_df = pd.DataFrame(columns=output_labels)
        predictions = []
        consts = self.train_df[self.labels].mean(axis=0, skipna=True)
        for id in self.test_ids:
            path = self.paths_dict[id]
            prediction = self.__generate_prediction(path, consts)
            prediction["Id"] = id
            # print(prediction)
            predictions.append(prediction.to_frame().T)
        output_df = pd.concat(predictions, axis=0)
        # Rearange columns
        cols = output_df.columns.to_list()
        output_df = output_df[cols[-1:] + cols[:-1]]

        # Convert rows to ints
        output_df = output_df.astype(int)
        
        # output_df is fully formed, now must save to file
        self.save_output(output_df)



    def __generate_prediction(self, path, consts) -> dict:
        """Generates a prediction from a path

        Args:
            path (string): filepath to image

        Returns:
            dict: result dict of 9 possibilities
        """
        # load image
        img = "image"

        # extract features
        features = self.get_features(path)

        # prediction = self.model.predict((img, features))
        
        # for simple case
        prediction = consts.to_numpy()

        prediction = self.format_prediction(prediction)
        
        return prediction
    

    def get_features(self, path) -> tuple:
        """Generate features from image path

        Args:
            path (str): path to image

        Returns:
            tuple: features
        """
        # This could just call functions from another file then format them
        return ()
    

    def format_prediction(self, prediction) -> dict:
        """Formats model prediction into tuple. Also calculates value of "No Finding"

        Args:
            prediction (tuple): A tuple of value

        Returns:
            dict: dictionary of prediction
        """
        if self.mode == 'round':
            output = [round(pred) for pred in prediction]
        
        if self.mode == 'prob':
            output = (int(math.floor(pred + random.random())) for pred in prediction)
        # turns values into series
        return pd.Series(output, index=self.labels)
    

    def save_output(self, output_df):
        """Saves the output_df to a csv

        Args:
            output_df (pd.DataFrame): output dataframe
        """
        time_string = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(self.output_folder, time_string + '.csv')
        output_df.to_csv(filepath, sep=',', header=True, index=False)



if __name__ == "__main__":
    prediction = Predictions()
    prediction.generate_predictions()