/*
	Copyright 2020 David Scholz

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

    	http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
 */
package org.pgrepds.deepl.nets;

import java.io.IOException;

import org.pgrepds.deepl.factories.DataSetIteratorFactory;
import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * A two layer neural network for the MNIST-Dataset.
 * 
 * @author David Scholz
 */
public class MnistMultiLayerNeuralNetwork extends AbstractMultiLayerNetwork {
	
	private DataSetIteratorFactory dataSetFactory = new DataSetIteratorFactory();
	
	private IMultiLayerConfiguration mnistMultiLayerConfig;
	
	private DataSetIterator mnistTrainingIterator;
	
	private DataSetIterator mnistEvaluationIterator;
	
	private boolean isUsingUI = false;
	
	public MnistMultiLayerNeuralNetwork(boolean useUI) throws MultiLayerNetworkException {
		super(useUI);
		this.isUsingUI = useUI;
		try {
			this.mnistTrainingIterator = dataSetFactory.createDataSetIterator(DataSetIteratorFactory.DataSetIteratorType.MNIST_TRAINING);
			this.mnistEvaluationIterator = dataSetFactory.createDataSetIterator(DataSetIteratorFactory.DataSetIteratorType.MNIST_EVAL);
		} catch (IOException e) {
			throw new MultiLayerNetworkException("Failed to initialize iterators. Aborted with message: " + e.getMessage());
		}
		
	}

	@Override
	public void eval() {
		
        eval = new Evaluation(MnistNetworkConstants.OUTPUT_CLASSES_COUNT);
        while (mnistEvaluationIterator.hasNext()) {
        	DataSet next = mnistEvaluationIterator.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output); 
        }
       
	}
	
	@Override
	public void train() {
		
        model.init();
        if (isUsingUI()) {
        	model.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage));
        } else {
        	model.setListeners(new ScoreIterationListener(1));
        }

        for (int i = 0; i < MnistNetworkConstants.EPOCH_COUNT; i++) {
        	model.fit(mnistTrainingIterator);
        }
	}

	@Override
	public void build(IMultiLayerConfiguration config) {
		
		this.mnistMultiLayerConfig = config;
        
        config.addLayer(0, new DenseLayer.Builder() 
                .nIn(MnistNetworkConstants.ROW_COUNT * MnistNetworkConstants.COLUMN_COUNT)
                .nOut(MnistNetworkConstants.ROW_COUNT * MnistNetworkConstants.COLUMN_COUNT)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .build());
        
        config.addLayer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) 
                .nIn(MnistNetworkConstants.ROW_COUNT * MnistNetworkConstants.COLUMN_COUNT)
                .nOut(MnistNetworkConstants.OUTPUT_CLASSES_COUNT)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build());
        
        model = new MultiLayerNetwork(config.createMultiLayerConfiguration());
	}

	@Override
	public IMultiLayerConfiguration getMultiLayerConfiguration() {

		return mnistMultiLayerConfig;
	}

	@Override
	public MultiLayerNetwork getMultiLayerModel() {

		return model;
	}
	
	private boolean isUsingUI() {
		
		return isUsingUI;
	}

}
