package org.pgrepds.deepl.nets;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.pgrepds.deepl.utils.LfwNetworkConstants;
import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.weights.ConvolutionalIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * An {@link AbstractMultiLayerNetworks} using convolutional layers with the
 * {@link LfwMultiLayerConfiguration} for the "labeled faces in the wild"
 * dataset (LFW).
 * 
 * @author David Scholz
 */
public class LfwMultiLayerNeuralNetwork extends AbstractMultiLayerNetwork {

	private IMultiLayerConfiguration lfwMultiLayerConfig;

	private boolean isUsingUI;

	private DataSetIterator iterator = null;
	
	private DataSet lfwDataSet = null;
	
	private SplitTestAndTrain train = null;
	
	private DataSet trainingInput = null;
	
	private List<INDArray> testList = new ArrayList<>();
	
	private List<INDArray> testLabels = new ArrayList<>();

	public LfwMultiLayerNeuralNetwork(boolean useUI) throws MultiLayerNetworkException {
		super(useUI);
		this.isUsingUI = useUI;

		/**
		 * The iterator for the lfw data set works different from the mnist dataset iterators. We have to use splitAndTrain in order to split the data.
		 */
		iterator = new LFWDataSetIterator(LfwNetworkConstants.BATCH_SIZE, LFWLoader.NUM_IMAGES,
				new int[] {LfwNetworkConstants.COLUMN_COUNT, LfwNetworkConstants.ROW_COUNT, LfwNetworkConstants.DEPTH}, 
				LFWLoader.NUM_LABELS, false, true, 1.0, new Random(LfwNetworkConstants.RANDOM_SEED));
	}

	@Override
	public IMultiLayerConfiguration getMultiLayerConfiguration() {

		return lfwMultiLayerConfig;
	}

	@Override
	public MultiLayerNetwork getMultiLayerModel() {

		return model;
	}

	@Override
	public void train() {
		model.init();
		// sets the listeners e.g. for gui and logging.
		if (isUsingUI) {
			model.setListeners(new ScoreIterationListener(1), new ConvolutionalIterationListener(1),new StatsListener(statsStorage));
		} else {
			model.setListeners(new ScoreIterationListener(1), new ConvolutionalIterationListener(1));
		}
		
		/**
		 * scale the images using deeplearning4j internal functions.
		 */
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        iterator.setPreProcessor(scaler);
        scaler.fit(iterator);
        
		while (iterator.hasNext()) {
			lfwDataSet = iterator.next();
			
			/**
			 * the normal scale function for the DataSet has some problems with rbg values.
			 */
			//lfwDataSet.scale();
			train = lfwDataSet.splitTestAndTrain(0.8);
			trainingInput = train.getTrain();
					
			testList.add(train.getTest().getFeatureMatrix());
			testLabels.add(train.getTest().getLabels());
			
			model.fit(trainingInput);
		}
	}

	@Override
	public void eval() {
		eval = new Evaluation(LfwNetworkConstants.OUTPUT_CLASS_COUNT);
		
		/**
		 * evaluate with the test labels and the test feature matrix.
		 */
		for (int i = 0; i < testList.size(); i++) {
			INDArray output = model.output(testList.get(i));
			eval.eval(testLabels.get(i), output);
		}
		
		INDArray o = model.output(testList.get(0));
		eval.eval(testLabels.get(0), o);
	}

	@Override
	public void build(IMultiLayerConfiguration config) {

		this.lfwMultiLayerConfig = config;

		config.addLayer(0, new ConvolutionLayer.Builder(3, 3)
				.nIn(LfwNetworkConstants.DEPTH)
				.stride(1, 1)
				.nOut(16)
				.build());

		config.addLayer(1, new SubsamplingLayer.Builder(SubsamplingLayer
				.PoolingType.MAX, new int[] { 2, 2 })
				.build());

		config.addLayer(2, new ConvolutionLayer.Builder(5, 5)
				.stride(1, 1)
				.nOut(32)
				.build());

		config.addLayer(3, new SubsamplingLayer.Builder(SubsamplingLayer
				.PoolingType.MAX, new int[] { 2, 2 })
				.build());

		config.addLayer(4, new DenseLayer.Builder()
				.nOut(128)
				.dropOut(0.5)
				.build());

		config.addLayer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
				.nOut(LfwNetworkConstants.OUTPUT_CLASS_COUNT)
				.activation(Activation.SOFTMAX)
				.build());

		model = new MultiLayerNetwork(config.createMultiLayerConfiguration());
	}

}
