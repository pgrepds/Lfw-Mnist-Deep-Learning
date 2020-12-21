package org.pgrepds.deepl.nets;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Basic API for a multilayered neural network to give so convenient methods.
 * 
 * @author David Scholz
 */
public interface IMultiLayerNeuralNetwork {
	
	/**
	 * Trains the network with the underlying model.
	 */
	void trainNetwork();
	
	/**
	 * Evaluates the model. Results are written to an log file. If no logging framework is defined 
	 * the result is written to stdout (very slow).
	 */
	void evalModel();
	
	/**
	 * 
	 * Builds the networks model.
	 * 
	 * @param config the {@link IMultiLayerConfiguration} of the model.
	 */
	void buildModel(IMultiLayerConfiguration config);
	
	/**
	 * 
	 * @return the {@link IMultiLayerConfiguration} of the underlying model.
	 */
	IMultiLayerConfiguration getMultiLayerConfiguration();
	
	/**
	 * 
	 * @return the {@link MultiLayerNetwork}, which represents the underlying model.
	 */
	MultiLayerNetwork getMultiLayerModel();

}
