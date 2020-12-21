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

import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * The MNIST dataset is special, meaning that it outputs 28x28 grayscale images in a row vector format (1x784). We must use flat convolutional networks.
 * 
 * @author David Scholz
 */
public class MnistConvolutionalMultiLayerConfiguration extends MultiLayerConfigurationImpl {

	public MnistConvolutionalMultiLayerConfiguration(double learningRate, Updater updater,
			OptimizationAlgorithm optimizationAlgorithm, boolean useBackprop) {
		super(learningRate, updater, optimizationAlgorithm, useBackprop);
	}
	
	public MnistConvolutionalMultiLayerConfiguration(double learningRate, Updater updater, OptimizationAlgorithm optimizationAlgorithm, 
			WeightInit weightInit, boolean useBackprop) {
		super(learningRate, updater, optimizationAlgorithm, weightInit, useBackprop);
	}
	
	@Override
	public MultiLayerConfiguration createMultiLayerConfiguration() {
		
		return listBuilder.setInputType(InputType.convolutionalFlat(MnistNetworkConstants.COLUMN_COUNT, MnistNetworkConstants.ROW_COUNT,
				MnistNetworkConstants.DEPTH)).pretrain(false).backprop(isUsingBackpropagation()).build();
	}

}
