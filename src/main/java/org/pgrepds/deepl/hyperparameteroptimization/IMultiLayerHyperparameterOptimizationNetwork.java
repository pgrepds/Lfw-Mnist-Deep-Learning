package org.pgrepds.deepl.hyperparameteroptimization;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * @author David Scholz
 */
public interface IMultiLayerHyperparameterOptimizationNetwork {
	
	/**
	 * Builds the model to optimize using {@link ParameterSpace}s, e.g. for the learning rate and the layer depth.
	 * 
	 * @param space the {@link IMultiLayerOptimizationSpace}.
	 */
	void build(IMultiLayerOptimizationSpace space);
	
	/**
	 * Starts the optimization.
	 */
	void startOptimization();
	
	/**
	 * Some basics statistics.
	 * 
	 * @return the {@link OptimizationResult} of the hyperparameter optimization.
	 */
	OptimizationResult getOptimizationResult();
	
	/**
	 * @return the local optimal {@link MultiLayerNetwork}.
	 */
	MultiLayerNetwork getOptimizedMultiLayerNetworkModel();

}
