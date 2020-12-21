package org.pgrepds.deepl.nets;

import org.apache.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

/**
 * Abstract basis class for all {@link IMultiLayerNeuralNetwork}s, for logging purposes.
 * 
 * @author David Scholz
 */
public abstract class AbstractMultiLayerNetwork implements IMultiLayerNeuralNetwork {
	
	protected static final Logger logger = Logger.getLogger(AbstractMultiLayerNetwork.class);
	
	protected Evaluation eval;
	
	protected MultiLayerNetwork model;
	
	protected StatsStorage statsStorage;
	
	/**
	 * http://localhost:9000/train
	 * @param useUI y
	 */
	public AbstractMultiLayerNetwork(boolean useUI) {
		if (useUI) {
			UIServer uiServer = UIServer.getInstance();
			statsStorage = new InMemoryStatsStorage();
			uiServer.attach(statsStorage);
		}
	}
	
	abstract void train();
	
	abstract void eval();
	
	abstract void build(IMultiLayerConfiguration config);
	
	@Override
	public void trainNetwork() {
		logger.info("Train model....");
		train();
	}

	@Override
	public void evalModel() {
		logger.info("Evaluate model....");
		eval();
		logger.info(eval.stats());
		// fallback just for testing.
		System.out.println(eval.stats());
	}

	@Override
	public void buildModel(IMultiLayerConfiguration config) {
		logger.info("Build model....");
		build(config);
	}

}
