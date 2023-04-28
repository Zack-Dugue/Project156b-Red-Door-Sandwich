import pandas as pd
import os
import time
import math
import random
from model import XRAYModel
from main import XrayModule
from torchvision.io import read_image
import argparse
import torch as th


NUM_CLASSES = 9 #this must be the same as in the main script

class Predictions:
    def __init__(self, model_path, mode):
        self.test_ids_filepath = os.path.join(os.getcwd(), 'data', 'student_labels', 'test_sample.csv')
        #this next line should be no longer neccesary
        self.train_labels_filepath = os.path.join(os.getcwd(), 'data', 'student_labels', 'train2023.csv')
        self.output_folder = "outputs/"
        self.model_path = model_path
        # Choose value probabalistically, or by rounding
        # Options: 'round', 'prob'
        self.mode = mode
        
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
        xray_model = XRAYModel(NUM_CLASSES)
        learning_rate = .001 #not sure if this matters as no learning will be done
        optimizer = th.optim.Adam(xray_model.parameters(),lr=learning_rate)
        self.model = XrayModule(xray_model, optimizer)

        self.model.load_state_dict(th.load(self.model_path, map_location=th.device('cpu')))
        self.model.eval()


    def generate_predictions(self):
        # output_labels = ("Id") + self.labels
        # output_df = pd.DataFrame(columns=output_labels)
        predictions = []
        consts = self.train_df[self.labels].mean(axis=0, skipna=True)
        for idx, id in enumerate(self.test_ids):
            path = self.paths_dict[id]
            prediction = self.__generate_prediction(path, consts)
            prediction["Id"] = id
            predictions.append(prediction.to_frame().T)
            print(idx)
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
        full_path = os.path.join(os.getcwd(), 'data', path)
        image = read_image(full_path)
        image = image.to(th.float32)/255.0
        image = image.to(th.float32)

        # make nan mask of 1s
        nan_mask = th.tensor([1 for _ in range(9)], dtype=th.float32)
        # extract features
        #  features = self.get_features(path)

        #prediction = self.model((image, nan_mask))
        prediction = self.model(image).detach().numpy()[0]
        
        # for simple case
        #prediction = consts.to_numpy()

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
            output = [round(pred*2-1) for pred in prediction]
        
        if self.mode == 'prob':
            output = [int(math.floor(pred*2-1 + random.random())) for pred in prediction]
       
        # recalculate index[0] which is No Finding
        output[0] = -1
        if output.count(1) == 0:
            output[0] = 1
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
    parser = argparse.ArgumentParser(
        prog="Predict",
        description='Produce a CSV of predictions for upload to eval.ai',
    )
    parser.add_argument('-p', '--path', help='Path to stored model (.pth)')
    parser.add_argument('-m', '--mode', choices=['round', 'prob'], required=False, default='round', help='How to select if passed value is between integers')
    
    args = parser.parse_args()
    path = args.path
    mode = args.mode    
    prediction = Predictions(path, mode)
    prediction.generate_predictions()