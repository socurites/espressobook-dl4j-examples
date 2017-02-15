package com.socurites.espressobook.example.chap3;


import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.datavec.api.io.converters.LabelWriterConverter;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.socurites.espressobook.example.util.Constants;

/**
 * IRIS classifier example using DenseLayer
 * 
 * @author socurites
 *
 */
public class SingleLayerPerceptronExample {
    private static Logger log = LoggerFactory.getLogger(SingleLayerPerceptronExample.class);
    
    /* 
     * # of input features. 
     */
    protected static final int numInput = 2;
    
    /* 
     * # of output classes.
     * Iris-setosa, Iris-versicolor, Iris-virginica
     */
    protected static final int numOutput = 1;
    
    /*
     * # of iterations.
     */
    protected static final int numIterations = 1; 
    
    /*
     * Learning rate.
     */
    protected static final double learningRate = 0.15;
    
    /*
     * # of epohcs.
     */
    protected static final int numEpochs = 25;
    
    /*
     * labels of output classes
     */
    protected static final List<String> labels = Arrays.asList(new String[] {"Iris-setosa", "Iris-versicolor", "Iris-virginica"});

    public static void main(String[] args) throws Exception {
    	SingleLayerPerceptronExample example = new SingleLayerPerceptronExample();
    	
    	// 1. Loads datasets for training and evaluation
    	INDArray inputX = Nd4j.create(new double[][] { {0.1, 0.3}, {0.8, 1.2}, {2.5, 0.3}, {1.2, 4.2}, {0.4, 0.8}, {3.2, 1.2}, {1.7, 0.1}, {3.1, 5.7}, {2.9, 0.5} });
    	INDArray inputY = Nd4j.create(new double[][] { {0}, {0}, {1}, {0}, {0}, {1}, {1}, {0}, {1} });
    	
    	// 2. Configures network
    	MultiLayerConfiguration conf = example.configureNetwork();
    	
    	// 3. Trains network using training datasets
    	MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(1));


        log.info(model.getLayer(0).paramTable().toString());
        
        log.info(inputX.getRow(0).toString());
        List<INDArray> feedForward = model.feedForward(inputX.getRow(0));
        
        log.info(feedForward.toString());
        
        model.fit(inputX, inputY);
        
        log.info(model.getLayer(0).paramTable().toString());
        
        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(inputX, inputY);
        }
    	
    }
    
    /**
     * Configures neural network
     * 
     * @return
     */
    private MultiLayerConfiguration configureNetwork() {
    	log.info("Build model....");
    	int layerIndex = 0;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(numIterations)
                .learningRate(learningRate)
                .list()
                .layer(layerIndex++, new OutputLayer.Builder(LossFunction.MSE)
                        .nIn(numInput)
                        .nOut(numOutput)
                        .activation(Activation.SIGMOID)
                        .build())
                .pretrain(false).backprop(true) 
                .build();
        
        return conf;
    }
}
