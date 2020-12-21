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
