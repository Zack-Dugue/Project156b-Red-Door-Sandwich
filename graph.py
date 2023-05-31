from matplotlib import pyplot as plt
import pandas as pd


points = [
    {"No Finding": 2.3260929162393493, "Average MSE": 2.0410470752713237, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 3.71409902020867, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.8617200674536254, "Fracture": 1.2018984667287347, "Support Devices": 5.200662955876503},
    {"No Finding": 2.3260929162393493, "Average MSE": 2.0410470752713237, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 3.71409902020867, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.8617200674536254, "Fracture": 1.2018984667287347, "Support Devices": 5.200662955876503},
    {"No Finding": 1.1736047647586647, "Average MSE": 1.1925741804867491, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 1.1128410767445505, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.758291174817313, "Fracture": 1.2018984667287347, "Support Devices": 1.0642251192456549},
    {"No Finding": 2.3260929162393493, "Average MSE": 2.0410470752713237, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 3.71409902020867, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.8617200674536254, "Fracture": 1.2018984667287347, "Support Devices": 5.200662955876503},
    {"No Finding": 1.484802689303742, "Average MSE": 1.4144584680950743, "Enlarged Cardiomediastinum": 1.3621270643369623, "Cardiomegaly": 1.1976488537056182, "Lung Opacity": 1.5384721095795237, "Pneumonia": 1.1449292104390876, "Pleural Effusion": 1.3156469652190161, "Pleural Other": 1.9066891512085438, "Fracture": 1.246809263528207, "Support Devices": 1.5464910690677491},
    {"No Finding": 1.1736047647586647, "Average MSE": 1.1925741804867491, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 1.1128410767445505, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.758291174817313, "Fracture": 1.2018984667287347, "Support Devices": 1.0642251192456549},
    {"No Finding": 1.1736047647586647, "Average MSE": 1.2042099309083345, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 1.1128410767445505, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.8617200674536254, "Fracture": 1.2018984667287347, "Support Devices": 1.0642251192456549},
    {"No Finding": 1.1688154297884013, "Average MSE": 0.9895719781298883, "Enlarged Cardiomediastinum": 0.904022225331005, "Cardiomegaly": 0.6959758275946109, "Lung Opacity": 1.0027195424526494, "Pneumonia": 1.0233438075606005, "Pleural Effusion": 0.5414314114482489, "Pleural Other": 1.5694210230466552, "Fracture": 0.9495425609031285, "Support Devices": 1.0592404582655817},
    {"No Finding": 8.397663476268873, "Average MSE": 2.725561106933559, "Enlarged Cardiomediastinum": 1.1692785483860975, "Cardiomegaly": 1.0382339733271966, "Lung Opacity": 3.71409902020867, "Pneumonia": 1.0115230045029697, "Pleural Effusion": 1.1900905499284282, "Pleural Other": 1.8617200674536254, "Fracture": 1.2018984667287347, "Support Devices": 5.200662955876503},
    {"No Finding": 0.9999560728500599, "Average MSE": 0.8182484167299048, "Enlarged Cardiomediastinum": 0.7455952237233222, "Cardiomegaly": 0.5726651980189706, "Lung Opacity": 0.8657631398385551, "Pneumonia": 0.9585175391007681, "Pleural Effusion": 0.46705279835609287, "Pleural Other": 0.9969136694554899, "Fracture": 0.8094780673736257, "Support Devices": 0.9650126473130098},
    {"No Finding": 8.397663476268873, "Average MSE": 1.6727127628935616, "Enlarged Cardiomediastinum": 0.7684947192839, "Cardiomegaly": 0.552668288757604, "Lung Opacity": 0.8478815584113731, "Pneumonia": 1.0325657311414638, "Pleural Effusion": 0.48081686756571673, "Pleural Other": 1.0931280785610789, "Fracture": 0.8238071198679815, "Support Devices": 0.9631547985524076},
    {"No Finding": 0.9999560728500599, "Average MSE": 0.838688935005305, "Enlarged Cardiomediastinum": 0.7684947192835665, "Cardiomegaly": 0.5526682887578427, "Lung Opacity": 0.8478815584113802, "Pneumonia": 1.0325657311417713, "Pleural Effusion": 0.480816867568883, "Pleural Other": 1.0931280785610833, "Fracture": 0.8238071198679624, "Support Devices": 0.9631547985524076},
    {"No Finding": 8.397663476268873, "Average MSE": 1.7324430062200318, "Enlarged Cardiomediastinum": 0.8681178417688816, "Cardiomegaly": 0.6029942943716691, "Lung Opacity": 0.9191292778810205, "Pneumonia": 1.1776090861113555, "Pleural Effusion": 0.4906679295043214, "Pleural Other": 1.124378528766379, "Fracture": 0.9125554912311608, "Support Devices": 1.002703519472262},
    {"No Finding": 0.7407923365120171, "Average MSE": 0.8434522182925598, "Enlarged Cardiomediastinum": 0.7881614354598556, "Cardiomegaly": 0.5743609483449351, "Lung Opacity": 0.8787394247184989, "Pneumonia": 1.1426693236606131, "Pleural Effusion": 0.485533589723095, "Pleural Other": 1.1646249839396883, "Fracture": 0.8643665488452477, "Support Devices": 0.9666167650319529},
    {"No Finding": 0.7735502611899092, "Average MSE": 0.8728939944602722, "Enlarged Cardiomediastinum": 0.8681178417688816, "Cardiomegaly": 0.6029942943716691, "Lung Opacity": 0.9191292778810205, "Pneumonia": 1.1776090861113555, "Pleural Effusion": 0.4906679295043214, "Pleural Other": 1.124378528766379, "Fracture": 0.9125554912311608, "Support Devices": 1.002703519472262},
    {"No Finding": 0.7368944816677131, "Average MSE": 0.8343044753833846, "Enlarged Cardiomediastinum": 0.8053286185758946, "Cardiomegaly": 0.5620876846314982, "Lung Opacity": 0.8623084747091577, "Pneumonia": 1.06639123544945, "Pleural Effusion": 0.4856534120406844, "Pleural Other": 1.2247811960937043, "Fracture": 0.8231574842003045, "Support Devices": 0.9565067556484682},
]

df = pd.DataFrame(points)
# print(df)
plt.figure(1)
plt.title("Loss")
plt.xlabel("Model Iteration")
plt.ylabel("MSE Loss")
#plt.plot([df["No Finding"], df["Average MSE"], df["Enlarged Cardiomediastinum"], df["Cardiomegaly"], df["Lung Opacity"], df["Pneumonia"], df["Pleural Effusion"], df["Pleural Other"], df["Fracture"], df["Support Devices"]])
for col in df.columns:
    plt.plot(df[col], label=col)
plt.axhline(y=1, color='k', linestyle='dashed')
plt.legend()
plt.show()