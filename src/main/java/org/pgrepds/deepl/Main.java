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
package org.pgrepds.deepl;

import java.io.IOException;

import org.pgrepds.deepl.utils.LfwNetworkConstants;
import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;

import org.pgrepds.deepl.hyperparameteroptimization.IMultiLayerHyperparameterOptimizationNetwork;
import org.pgrepds.deepl.hyperparameteroptimization.IMultiLayerOptimizationSpace;
import org.pgrepds.deepl.hyperparameteroptimization.MnistMultiLayerOptimizationSpace;
import org.pgrepds.deepl.hyperparameteroptimization.MnistSelfOptimizingMultiLayerNeuralNetwork;
import org.pgrepds.deepl.nets.IMultiLayerConfiguration;
import org.pgrepds.deepl.nets.IMultiLayerNeuralNetwork;
import org.pgrepds.deepl.nets.LfwMultiLayerConfiguration;
import org.pgrepds.deepl.nets.LfwMultiLayerNeuralNetwork;
import org.pgrepds.deepl.nets.MnistConvolutionalMultiLayerConfiguration;
import org.pgrepds.deepl.nets.MnistConvolutionalMultiLayerNeuralNetwork;
import org.pgrepds.deepl.nets.MnistMultiLayerNeuralNetwork;
import org.pgrepds.deepl.nets.MultiLayerConfigurationImpl;
import org.pgrepds.deepl.nets.MultiLayerNetworkException;

/**
 * The main application.
 * 
 * @author David Scholz
 */
public class Main {
    
	public static void main( String[] args ) throws IOException {
		
		// +++++++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++++++++++++++++++++++
		// Multi layer network for the MNIST-Dataset using a self-picked learning rate. 
		
		IMultiLayerConfiguration mnistMultiLayerNetworkConfig = new MultiLayerConfigurationImpl(MnistNetworkConstants.LEARNING_RATE,
				Updater.NESTEROVS, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, true);
		
		IMultiLayerNeuralNetwork mnistMultiLayerNetwork;
		try {
			mnistMultiLayerNetwork = new MnistMultiLayerNeuralNetwork(true);
			mnistMultiLayerNetwork.buildModel(mnistMultiLayerNetworkConfig);
			mnistMultiLayerNetwork.trainNetwork();
			mnistMultiLayerNetwork.evalModel();
		} catch (MultiLayerNetworkException e) {
			e.printStackTrace();
			return;
		}
		
		//++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		// +++++++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++++++++++++++++++++++
	    // Multi layer network for the MNIST-Dataset using hyperparemter optimization for the learning rate.
		
		IMultiLayerOptimizationSpace space = new MnistMultiLayerOptimizationSpace(WeightInit.XAVIER, true);
		IMultiLayerHyperparameterOptimizationNetwork optimizationNetwork = new MnistSelfOptimizingMultiLayerNeuralNetwork();
		optimizationNetwork.build(space);
		optimizationNetwork.startOptimization();
		
		// for testing purposes printing the optimal model as json string.
		System.out.println(optimizationNetwork.getOptimizedMultiLayerNetworkModel().getLayerWiseConfigurations().toJson());
		
		//++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		// +++++++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++++++++++++++++++++++
	    // Multi layer network for the MNIST-Dataset using convolutional layers with a self-picked learning rate
				
		IMultiLayerConfiguration mnistConvolMultiLayerNetworkConfig = new MnistConvolutionalMultiLayerConfiguration(
				MnistNetworkConstants.LEARNING_RATE, 
				Updater.NESTEROVS, 
				OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 
				WeightInit.XAVIER, true);
		
		IMultiLayerNeuralNetwork mnistConvMultiLayerNetwork;
		
		try {
			mnistConvMultiLayerNetwork = new MnistConvolutionalMultiLayerNeuralNetwork(false);
			mnistConvMultiLayerNetwork.buildModel(mnistConvolMultiLayerNetworkConfig);
			mnistConvMultiLayerNetwork.trainNetwork();
			mnistConvMultiLayerNetwork.evalModel();
		} catch (MultiLayerNetworkException e) {
			e.printStackTrace();
			return;
		}
				
		//++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		
		// +++++++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++++++++++++++++++++++
	    // Multi layer network for the Lfw-Dataset 
		
		IMultiLayerConfiguration lfwMultiLayerNetworkConfig = new LfwMultiLayerConfiguration(
				LfwNetworkConstants.LEARNING_RATE,
				Updater.ADAGRAD, 
				OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 
				WeightInit.XAVIER, true);
		
		IMultiLayerNeuralNetwork lfwConvMultiLayerNetwork;
		
		try {
			lfwConvMultiLayerNetwork = new LfwMultiLayerNeuralNetwork(false);
			lfwConvMultiLayerNetwork.buildModel(lfwMultiLayerNetworkConfig);
			lfwConvMultiLayerNetwork.trainNetwork();
			lfwConvMultiLayerNetwork.evalModel();
		} catch (MultiLayerNetworkException e) {
			e.printStackTrace();
		}
		//++++++++++++++++++++++++++++++++++ END ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    }	
    
}
