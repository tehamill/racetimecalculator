import run_config
import matplotlib.pyplot as plt
import predict_time


def main():
    config = run_config.RunConfig
    #predict time takes config,datafile. datafile should have weather included in it
    runmod = predict_time.PredictTime(config,"edawithWeather.csv")
    runmod.run_model() #run model, printoutput
    runmod.make_plots()

    return





if __name__ == "__main__":
    main()
