import run_config
import matplotlib.pyplot as plt
import predict_time


def main():
    config = run_config.RunConfig
    #predict time takes config,datafile. datafile should have weather included in it
    
    race_model = predict_time.PredictTime(config)
    race_model.clean_data()
    race_model.train_model() #run model, printoutput
    runmod.make_plots()
    #to predict new data point
    #race_model.model.predict(data)
    #where data has same form as race_model.Xtrain
    return





if __name__ == "__main__":
    main()
