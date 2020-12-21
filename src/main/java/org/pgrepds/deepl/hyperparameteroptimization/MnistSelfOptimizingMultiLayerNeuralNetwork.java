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
package org.pgrepds.deepl.hyperparameteroptimization;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.pgrepds.deepl.factories.DataSetIteratorFactory;
import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Self optimizing neural network, which uses hyperparameter optimization by using random search methods 
 * to find the optimal learning rate and layer size.
 * 
 * @author David Scholz
 */
public class MnistSelfOptimizingMultiLayerNeuralNetwork implements IMultiLayerHyperparameterOptimizationNetwork {
	
	private MultiLayerSpace hyperParamSpace;
	
	private MultiLayerNetwork optimizedNetwork;
	
	private OptimizationResult optimizationResult;
	
	public MnistSelfOptimizingMultiLayerNeuralNetwork() {
		
	}

	@Override
	public void build(IMultiLayerOptimizationSpace space) {
		
		// basically the same as in the normal mnist neural network setup.
		space.addLayer(new DenseLayerSpace.Builder()
				.nIn(MnistNetworkConstants.COLUMN_COUNT * MnistNetworkConstants.ROW_COUNT)
				.activation(Activation.LEAKYRELU)
				.nOut(space.getLayerDepthParameterSpace())
				.build());
		
		space.addLayer(new OutputLayerSpace.Builder()
				.nOut(MnistNetworkConstants.OUTPUT_CLASSES_COUNT)
				.activation(Activation.SOFTMAX)
				.lossFunction(LossFunctions.LossFunction.MCXENT)
				.build());
		
		hyperParamSpace = space.createMultiLayerSpace();
	}

	@Override
	public OptimizationResult getOptimizationResult() {
	
		return optimizationResult;
	}

	@Override
	public MultiLayerNetwork getOptimizedMultiLayerNetworkModel() {

		return optimizedNetwork;
	}

	private DataProvider getDataProvider() {
		
		int epochs = 2;
		
		/**
		 * One needs own data provider, basically the same as the normale mnist dataset iterator but implemented as an DataProvider.
		 */
		return new DataProvider() {

			private static final long serialVersionUID = 6515884631182991944L;
			
			private DataSetIteratorFactory factory = new DataSetIteratorFactory();

			@Override
			public Object trainData(Map<String, Object> arg0) {
				
				try {
					
					return new MultipleEpochsIterator(epochs, factory.createDataSetIterator(DataSetIteratorFactory.DataSetIteratorType.MNIST_TRAINING));
					
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
			
			@Override
			public Object testData(Map<String, Object> arg0) {

				try {
					
					return factory.createDataSetIterator(DataSetIteratorFactory.DataSetIteratorType.MNIST_EVAL);
					
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
			
			@Override
			public Class<?> getDataType() {
	
				return DataSetIterator.class;
			}
		};
	}
	
	/**
	 * Saves to file.
	 */
	private ResultSaver getResultSaver() {
		String baseDirPath = "arbiter/";
		File file = new File(baseDirPath);
		if (file.exists()) 
			file.delete();
		
		file.mkdir();
		ResultSaver resultSaver = new FileModelSaver(baseDirPath);
		
		return resultSaver;
	}
	
	private TerminationCondition[] getTerminationConditions() {
		
		// break after 10 minutes or 10 candidates.
		TerminationCondition[] conditions = {
				new MaxTimeCondition(10, TimeUnit.MINUTES),
				new MaxCandidatesCondition(10)
		};
		
		return conditions;
	}
	
	@Override
	public void startOptimization() {

		CandidateGenerator generator = new RandomSearchGenerator(hyperParamSpace, null);
		DataProvider iterator = getDataProvider();
		ResultSaver resultSaver = getResultSaver();
		ScoreFunction scoreFunction = new TestSetAccuracyScoreFunction();
		
		// optimization config.
		OptimizationConfiguration config = new OptimizationConfiguration.Builder()
				.candidateGenerator(generator)
				.dataProvider(iterator)
				.modelSaver(resultSaver)
				.scoreFunction(scoreFunction)
				.terminationConditions(getTerminationConditions())
				.build();
		
		IOptimizationRunner optimizationRunner = new LocalOptimizationRunner(config, new MultiLayerNetworkTaskCreator());
		
		optimizationRunner.execute();
		
		int bestResultIdx = optimizationRunner.bestScoreCandidateIndex();
		List<ResultReference> allResults = optimizationRunner.getResults();
		
		try {
			optimizationResult = allResults.get(bestResultIdx).getResult();
			optimizedNetwork = (MultiLayerNetwork) optimizationResult.getResult();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


}
