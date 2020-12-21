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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * An {@link MnistMultiLayerNeuralNetwork} using convolutional layer. Trains with the MNIST dataset.
 * 
 * @author David Scholz
 */
public class MnistConvolutionalMultiLayerNeuralNetwork extends MnistMultiLayerNeuralNetwork {

	public MnistConvolutionalMultiLayerNeuralNetwork(boolean useUI) throws MultiLayerNetworkException {
		super(useUI);
	}

	@Override
	public void build(IMultiLayerConfiguration config) {
		
		config.addLayer(0, new ConvolutionLayer.Builder(5, 5)
				.nIn(1) 
				.stride(1, 1)
				.nOut(16)
				.activation(Activation.IDENTITY) 
				.build());
		
		config.addLayer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
				.kernelSize(2, 2)
				.stride(2, 2)
				.build());
		
		config.addLayer(2, new ConvolutionLayer.Builder(5, 5)
				.stride(1, 1)
				.nOut(32)
				.activation(Activation.IDENTITY)
				.build());
		
		config.addLayer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
				.kernelSize(2, 2)
				.stride(2, 2)
				.build());
		
		config.addLayer(4, new DenseLayer.Builder()
				.activation(Activation.RELU)
				.nOut(500)
				.build());
		
		config.addLayer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				.nOut(MnistNetworkConstants.OUTPUT_CLASSES_COUNT)
				.activation(Activation.SOFTMAX)
				.build());
		
		 model = new MultiLayerNetwork(config.createMultiLayerConfiguration());
	}

}
