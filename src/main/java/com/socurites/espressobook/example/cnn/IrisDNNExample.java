package com.socurites.espressobook.example.cnn;


import java.io.IOException;
import java.util.Arrays;

import org.datavec.api.io.converters.LabelWriterConverter;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * @author socurites
 *
 */
public class IrisDNNExample {
    private static Logger log = LoggerFactory.getLogger(IrisDNNExample.class);
    
    /* random number seedf for reproducibility. */
    public static final int RNG_SEED = 123;
    
    /* batch size for each epoch. */
    public static final int BATCH_SIZE = 5;
    
    /* # of input features. */
    public static final int INPUT_NUM = 4;
    
    /* # of output classes */
    public static final int OUTPUT_NUM = 3;
    
    public static final String[] labels = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    public static void main(String[] args) throws Exception {
    	IrisDNNExample example = new IrisDNNExample();
    	
    	DataSetIterator trainDataSetIterator 	= example.loadDataSet("data/iris/iris_train.data");
    	DataSetIterator testDataSetIterator 	= example.loadDataSet("data/iris/iris_test.data");
    	MultiLayerConfiguration conf 			= example.configureNetwork();
    	MultiLayerNetwork model					= example.train(conf, trainDataSetIterator);
    											  example.evaluate(model, testDataSetIterator);
    }
    
    private DataSetIterator loadDataSet(String dataFilePath) throws IOException, InterruptedException {
    	CSVRecordReader recordReader = new CSVRecordReader();
    	recordReader.initialize(new FileSplit(new ClassPathResource(dataFilePath).getFile()));
    	
    	LabelWriterConverter labelConverter = new LabelWriterConverter(Arrays.asList(labels));
    	
    	RecordReaderDataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, labelConverter, BATCH_SIZE, 4, 3);
    	
    	return dataSetIterator;
    }
    
    private MultiLayerConfiguration configureNetwork() {
    	log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        		// include a random seed for reproducibility
                .seed(RNG_SEED) 
                // use stochastic gradient descent as an optimization algorithm
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(5)
                // specify the learning rate
                .learningRate(0.01)
                // specify the rate of change of the learning rate.
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(INPUT_NUM)
                        .nOut(10)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.XENT) //create hidden layer
                        .nIn(10)
                        .nOut(OUTPUT_NUM)
                        .activation(Activation.SOFTMAX)
                        .build())
                // use backpropagation to adjust weights
                .pretrain(false).backprop(true) 
                .build();
        
        return conf;
    }
    
    private MultiLayerNetwork train(MultiLayerConfiguration conf, DataSetIterator trainSetIterator) throws IOException {
        int numEpochs = 2; // number of epochs to perform


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(trainSetIterator);
        }

        return model;
    }

    private void evaluate(MultiLayerNetwork model, DataSetIterator testSetIterator) {
    	log.info("Evaluate model....");
//        Evaluation eval = new Evaluation(OUTPUT_NUM); //create an evaluation object with 10 possible classes
        Evaluation eval = new Evaluation(Arrays.asList(labels));
        while(testSetIterator.hasNext()){
            DataSet t = testSetIterator.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);
            
            eval.eval(lables, predicted);
            
            System.out.println(lables);
            System.out.println(predicted);

        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
    }
}
