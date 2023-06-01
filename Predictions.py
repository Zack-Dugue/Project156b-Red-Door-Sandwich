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
import pytorch_lightning as pl
from XRAYdataLoader import make_dataloader


NUM_CLASSES = 9 #this must be the same as in the main script

class Predictions:
    def __init__(self, model_path, mode, phase):
        self.test_ids_filepath = os.path.join(os.getcwd(), 'data', 'student_labels', 'test_ids.csv')
        self.sol_ids_filepath = os.path.join(os.getcwd(), 'data', 'labels2023', 'solution2023_path_to_id.csv')
        # self.test_ids_filepath = os.path.join(os.getcwd(), 'data', 'student_labels', 'test_ids.csv')
        #this next line should be no longer neccesary
        self.train_labels_filepath = os.path.join(os.getcwd(), 'data', 'student_labels', 'train_sample.csv')
        self.output_folder = "outputs/"
        self.model_path = model_path
        # Choose value probabalistically, or by rounding
        # Options: 'round', 'prob'
        self.mode = mode
        self.phase = phase
        
        self.ingest_test_csv()
        self.ingest_train_csv()
        self.load_model()

    def generate_bulk_predictions(self):
        self.bulk_predict()
        self.calculate_no_findings()
        self.apply_bulk_predict()


    def ingest_test_csv(self):
        if self.phase == 'dev':
            df = pd.read_csv(self.test_ids_filepath, sep=',', header='infer')
        if self.phase == 'solution':
            df = pd.read_csv(self.sol_ids_filepath, sep=',', header='infer')
        self.test_ids = df["Id"]
        self.paths_dict = dict(zip(df["Id"], df["Path"]))

    def ingest_train_csv(self):
        df = pd.read_csv(self.train_labels_filepath, sep=',', header='infer')
        self.train_df = df
        self.labels = df.columns[-9:]

    
    def load_model(self):
        xray_model = XRAYModel(NUM_CLASSES)
        # learning_rate = .001 #not sure if this matters as no learning will be done
        # optimizer = th.optim.Adam(xray_model.parameters(),lr=learning_rate)
        # self.model = XrayModule(xray_model, optimizer)
        checkpoint = th.load(self.model_path, map_location=th.device('cuda'))
        # checkpoint = th.load(self.model_path, map_location='cpu')
        self.model = XrayModule(xray_model)
        self.model.load_state_dict(checkpoint['state_dict'])
        # self.model = XrayModule.load_from_checkpoint(checkpoint[''])
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
        if self.mode == 'none' or self.mode == 'None':
            output = [pred*2-1 for pred in prediction]
       
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
        print('Finished outputting to:')
        print(os.path.abspath(filepath))


    def bulk_predict(self):
        trainer = pl.Trainer(accelerator='cuda', devices=1, strategy="auto", num_nodes=1)
        # trainer = pl.Trainer(accelerator='auto')
        data_loader = make_dataloader(self.test_ids_filepath, batch_size=128, num_dataloaders=4, train=False)
        self.predictions = trainer.predict(self.model, data_loader)
        self.predictions = pd.DataFrame(th.cat(self.predictions).numpy())
        
    def calculate_no_findings(self):
        # round values correctly
        if self.mode == 'round':
            df = self.predictions.transform(lambda x: round(2*x-1))
        if self.mode == 'prob':
            df = self.predictions.transform(lambda x: int(math.floor(x*2-1 + random.random())))
        if self.mode == 'None' or self.mode == 'none':
            df = self.predictions.transform(lambda x: 2*x-1)
        # calculate no findings
        # df[0] = df.apply(lambda x: -1 if 1 in x[1:].unique() else 1, axis=1)
        self.prediction_df = df

    def apply_bulk_predict(self):
        output_df = pd.concat((self.test_ids, self.prediction_df), axis=1)
        # rename columns
        column_names = ['Id'] + self.labels.to_list()
        output_df = output_df.rename(columns=dict(zip(output_df.columns, column_names)))
        self.save_output(output_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Predict",
        description='Produce a CSV of predictions for upload to eval.ai',
    )
    parser.add_argument('-p', '--path', required=True, help='Path to stored model (.pth)')
    parser.add_argument('-m', '--mode', choices=['round', 'prob', 'none'], required=False, default='round', help='How to select if passed value is between integers')
    parser.add_argument('-p', '--solution_phase', choices=['dev', 'solution'], required=False, default='solution', help='Which solution phase to predict on', type=str)
    
    args = parser.parse_args()
    path = args.path
    mode = args.mode
    phase = args.solution_phase
    prediction = Predictions(path, mode, phase)

    prediction.generate_bulk_predictions()
