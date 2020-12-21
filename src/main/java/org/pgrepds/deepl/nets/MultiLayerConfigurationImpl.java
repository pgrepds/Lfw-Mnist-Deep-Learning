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

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * Neural network configuration for multiple layers. This implementation has the only purpose to garantee easier access to the networks parameters.
 * 
 * @author David Scholz
 */
public class MultiLayerConfigurationImpl implements IMultiLayerConfiguration {
	
	protected ListBuilder listBuilder;
	
	private double learningRate = 0;
	
	private Updater updater = null;
	
	private boolean isUsingBackprop = false;
	
	private OptimizationAlgorithm optimizationAlgorithm = null;
	
	private int randomSeed = 42;
	
	private WeightInit weightInit;
	
	public MultiLayerConfigurationImpl(double learningRate, Updater updater, 
			OptimizationAlgorithm optimizationAlgorithm, boolean useBackprop) {
		this.learningRate = learningRate;
		this.updater = updater;
		this.isUsingBackprop = useBackprop;
		this.optimizationAlgorithm = optimizationAlgorithm;
		validateUpdater(updater);
		createStandardListBuilder();
	}
	
	public MultiLayerConfigurationImpl(double learningRate, Updater updater, OptimizationAlgorithm optimizationAlgorithm, 
			WeightInit weightInit, boolean useBackprop) {
		this.learningRate = learningRate;
		this.updater = updater;
		this.isUsingBackprop = useBackprop;
		this.optimizationAlgorithm = optimizationAlgorithm;
		this.weightInit = weightInit;
		createListBuilder();
	}
	
	private void createListBuilder() {
		
		listBuilder = new NeuralNetConfiguration.Builder()
				.seed(randomSeed)
				.optimizationAlgo(optimizationAlgorithm)
				.iterations(1)
				.learningRate(learningRate)
				.updater(updater)
				.regularization(true).l2(1e-4)
				.weightInit(weightInit)
				.list();
	}
	
	private void createStandardListBuilder() {
		
		// standard configuration for simple networks
		listBuilder = new NeuralNetConfiguration.Builder()
                .seed(randomSeed) 
                .optimizationAlgo(optimizationAlgorithm)
                .iterations(1)
                .learningRate(learningRate)
                .updater(updater)
                .regularization(true).l2(1e-4)
                .list();
	}
	
	private void validateUpdater(Updater updater) {
		if (updater == null) {
			throw new IllegalArgumentException("Updater cannot be null!");
		}
	}

	@Override
	public MultiLayerConfiguration createMultiLayerConfiguration() {

		return listBuilder.pretrain(false).backprop(isUsingBackprop).build();
	}
	
	@Override
	public void addLayer(int pos, Layer layer) {
		
		listBuilder.layer(pos, layer);
	}

	@Override
	public String getUpdaterName() {
		
		return updater.name();
	}

	@Override
	public double getLearningRate() {
		
		return learningRate;
	}

	@Override
	public boolean isUsingBackpropagation() {

		return isUsingBackprop;
	}

	@Override
	public int getRandomSeed() {

		return randomSeed;
	}

	@Override
	public WeightInit getWeightInit() {
		
		return weightInit;
	}

}
