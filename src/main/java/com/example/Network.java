package com.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network {
    Neuron first = new Neuron();
    Neuron second = new Neuron();
    Neuron third = new Neuron();
    Neuron fourth = new Neuron();
    Neuron out = new Neuron();


   
     /*
     * Data flows:
     * inputs → hidden layer → output layer
     */
    public double predict(double x1, double x2) {

        // Pass inputs into hidden neurons
        double thirdPrediction = third.predict(first.predict(x1, x2), second.predict(x1,x2));
        double fourthPrediction = fourth.predict(first.predict(x1, x2), second.predict(x1,x2));

        // Hidden outputs become inputs to final neuron
        return out.predict(thirdPrediction, fourthPrediction);
        
    }

      /**
     * Reads CSV file into a 2D list of strings
     */
    public static List<List<String>> readCSV(String resourceName) {
        List<List<String>> data = new ArrayList<>();

        BufferedReader br = null;
        try {
            InputStream is = Network.class.getResourceAsStream("/" + resourceName);
            if (is != null) {
                br = new BufferedReader(new InputStreamReader(is));
            } else {
                br = new BufferedReader(new FileReader(resourceName));
            }

            String line;
            while ((line = br.readLine()) != null) {
                data.add(Arrays.asList(line.split(",")));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try { br.close(); } catch (IOException ignored) {}
            }
        }

        return data;
    }

    
    /**
     * TRAINING FUNCTION
     * 
     * Uses gradient descent + backpropagation
     * to adjust weights and reduce error.
     */
    public void train(ArrayList<double[]> data, ArrayList<Double> answers) {

        double lr = 0.1; // learning rate

        // Repeat many times so the network can improve
        for (int epoch = 0; epoch < 10000; epoch++) {

            double totalLoss = 0;

            // Go through each training example
            for (int i = 0; i < data.size(); i++) {

                double x1 = data.get(i)[0];
                double x2 = data.get(i)[1]; 
                double correctAnswer = answers.get(i);   

                // =========================
                // 1. FORWARD PASS
                // =========================

                // Hidden neuron 3
                double z3 = third.rawValue(first.rawValue(x1, x2), second.rawValue(x1,x2));     
                double thirdOut = third.predict(first.predict(x1, x2), second.predict(x1,x2)); 

                // Hidden neuron 4
                double z4 =  fourth.rawValue(first.rawValue(x1, x2), second.rawValue(x1,x2));  
                double fourthOut = fourth.predict(first.predict(x1, x2), second.predict(x1,x2)); 

                // Output neuron
                double z5 =  out.rawValue(thirdOut, fourthOut); 
                double pred = out.predict(thirdOut, fourthOut);

                // =========================
                // 2. LOSS (error)
                // =========================

                double error = pred - correctAnswer;

                // Mean Squared Error contribution
                totalLoss += error * error;

                // =========================
                // 3. BACKPROPAGATION
                // =========================
                // We compute how much each weight contributed to the error.

                // ----- Output neuron -----

                double predictionLoss = 2 * error;
                double predDerivitive = Neuron.sigmoidDerivative(z5);

                // Gradients for output neuron weights
                double grad_out_w1 = predictionLoss * predDerivitive * thirdOut;
                double grad_out_w2 = predictionLoss * predDerivitive * fourthOut;
                double grad_out_b  = predictionLoss * predDerivitive;

                // ----- Hidden neurons -----
                // Activation derivatives
                double dz1 = Neuron.sigmoidDerivative(z3);
                double dz2 = Neuron.sigmoidDerivative(z4);

                // Gradients for hidden neuron 1
                double grad_h1_w1 = predictionLoss * predDerivitive * out.w1 * dz1 * x1;
                double grad_h1_w2 = predictionLoss * predDerivitive * out.w1 * dz1 * x2;
                double grad_h1_b  = predictionLoss * predDerivitive * out.w1 * dz1;

                // Gradients for hidden neuron 2
                double grad_h2_w1 = predictionLoss * predDerivitive * out.w2 * dz2 * x1;
                double grad_h2_w2 = predictionLoss * predDerivitive * out.w2 * dz2 * x2;
                double grad_h2_b  = predictionLoss * predDerivitive * out.w2 * dz2;

                // =========================
                // 4. UPDATE WEIGHTS
                // =========================
                // Each neuron adjusts its own weights

                out.update(grad_out_w1, grad_out_w2, grad_out_b, lr);

                first.update(grad_h1_w1, grad_h1_w2, grad_h1_b, lr);
                second.update(grad_h2_w1, grad_h2_w2, grad_h2_b, lr);
            }

            // Print progress every 1000 epochs
            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch +
                        " Loss: " + (totalLoss / data.size()));
            }
        }
    }


    public static void main(String[] args) 
    {   
        List<List<String>> completeData = readCSV("pokemon_complete.csv"); 
        ArrayList<double[]> data = new ArrayList<double[]>();
        ArrayList<Double> answers = new ArrayList<Double>();
        for (int i = 1; i < completeData.size(); i++)
        {
            data.add(new double[]{Double.parseDouble(completeData.get(i).get(4)) / 153, Double.parseDouble(completeData.get(i).get(5)) / 171.5});
            answers.add(Double.parseDouble(completeData.get(i).get(11)) / 10.09);
        }
        System.out.println(answers.get(0));
        
        Network network = new Network();
        network.train(data, answers);
        //Try making some predictions:
        System.out.println("Should give no "+network.predict(75/153, 80/171));
    }


}
