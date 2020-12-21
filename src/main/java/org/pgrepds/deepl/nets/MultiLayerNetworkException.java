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

/**
 * Network specific exception.
 * 
 * 
 * @author David Scholz
 */
public class MultiLayerNetworkException extends Exception {

	private static final long serialVersionUID = -4438432871071761159L;
	
	public MultiLayerNetworkException(final String msg) {
		super(msg);
	}
	
	public MultiLayerNetworkException(final String msg, final Exception cause) {
		super(msg, cause);
	}

}
