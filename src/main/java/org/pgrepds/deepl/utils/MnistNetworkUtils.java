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
package org.pgrepds.deepl.utils;

/**
 * Basic utility class. The main purpose of this class was to determine configuration problems with platform 
 * specific libraries, which prevented deeplearning4j to work.
 * 
 * @author David Scholz
 */
public final class MnistNetworkUtils {
	
	
	public MnistNetworkUtils() {
		
	}
	
	private static void load(String name) {
	  
		try{
    	  
          System.out.println("Trying to load: "+name);
          System.loadLibrary(name);
          
		}catch (Throwable e){
          System.out.println("Failed: "+e.getMessage());
          return;
		}
      
      System.out.println("Success");
	}

	public static void testNativeLibFailures() {
      load("libwinpthread-1");
      load("libstdc++-6");
      load("libquadmath-0");
      load("libopenblas");
      load("libgomp-1");
      load("libgfortran-3");
      load("libgcc_s_seh-1");
	}

}
