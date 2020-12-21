package org.pgrepds.deepl.hyperparameteroptimization;

import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * {@link IMultiLayerOptimizationSpace} for the mnist dataset, optimizing the layer depth and the learning rate.
 * 
 * @author David Scholz
 */
public class MnistMultiLayerOptimizationSpace implements IMultiLayerOptimizationSpace {
	
	private ParameterSpace<Double> learningRateHyperparameterSpace = new ContinuousParameterSpace(MnistNetworkConstants.HYPERPARAMTER_LEARNINGRATE_LOWER_BOUND,
			MnistNetworkConstants.HYPERPARAMTER_LEARNINGRATE_UPPER_BOUND);
	
	private ParameterSpace<Integer> layerDepthHyperparameterSpace = new IntegerParameterSpace(MnistNetworkConstants.HYPERPARAMTER_LAYERDEPTH_LOWER_BOUND,
			MnistNetworkConstants.HYPERPARAMTER_LAYERDEPTH_UPPER_BOUND);
	
	private WeightInit weightInit = null;
	
	private boolean useRegularization = false;
	
	private MultiLayerSpace.Builder multiLayerSpace;
	
	public MnistMultiLayerOptimizationSpace(WeightInit weightInit, boolean useRegularization) {
		this.weightInit = weightInit;
		this.useRegularization = useRegularization;
		multiLayerSpace = new MultiLayerSpace.Builder();
		
		multiLayerSpace.weightInit(weightInit).regularization(useRegularization).l2(MnistNetworkConstants.HYPERPARAMTER_LEARNINGRATE_LOWER_BOUND)
		.learningRate(learningRateHyperparameterSpace);
	}

	@Override
	public void addLayer(LayerSpace<?> layer) {
		
		multiLayerSpace.addLayer(layer);
		
	}

	@Override
	public MultiLayerSpace createMultiLayerSpace() {

		return multiLayerSpace.build();
	}

	public WeightInit getWeightInit() {
		
		return weightInit;
	}
	
	public boolean isUsingRegularization() {
		
		return useRegularization;
	}
	
	@Override
	public ParameterSpace<Integer> getLayerDepthParameterSpace() {
		
		return layerDepthHyperparameterSpace;
	}
}
