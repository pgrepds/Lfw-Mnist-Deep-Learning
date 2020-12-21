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
package org.pgrepds.deepl.iterators;

import java.io.IOException;

import org.pgrepds.deepl.utils.MnistNetworkConstants;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;

/**
 * Purpose of this class is to have all iterator constants at one place. Makes testing much easier.
 * 
 * @author David Scholz
 */
public class MnistTrainingDataSetIterator extends MnistDataSetIterator {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 8461657372724656299L;

	public MnistTrainingDataSetIterator() throws IOException {
		super(MnistNetworkConstants.BATCH_SIZE, true, MnistNetworkConstants.RANDOM_SEED);
	}

}
