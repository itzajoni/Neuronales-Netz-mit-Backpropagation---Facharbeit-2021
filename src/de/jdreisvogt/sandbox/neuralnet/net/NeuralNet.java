/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.net;

import de.jdreisvogt.sandbox.neuralnet.layer.HiddenLayer;
import de.jdreisvogt.sandbox.neuralnet.layer.Layer;
import de.jdreisvogt.sandbox.neuralnet.layer.OutputLayer;
import de.jdreisvogt.sandbox.neuralnet.neuron.TooManyInputsException;

import java.util.ArrayList;

/**
 * Repräsentiert ein KNN
 */
public class NeuralNet
{
    private final ArrayList<HiddenLayer> hiddenLayers = new ArrayList<>();
    private OutputLayer outputLayer;
    private double learningRate;
    private double tolerance;
    private double standardBias = 0;
    private final int inputSize;

    /**
     * Repräsentiert das Ergebnis eines Trainingsdurchgangs
     */
    public static class NeuralNetTrainingResult
    {
        public static class Result
        {
            public int correct = 0;
            public int wrong = 0;
            public int unclassified = 0;
            public int[] total_per_label = new int[10];
            public int[] correct_per_label = new int[10];
            public double[] absolute_output_error;
            public double[] relative_output_error;
        }

        public Result[] trainingResults;
        public Result[] testResults;
        public int time;

        public static String arrayToString (double[] stats)
        {
            StringBuilder result = new StringBuilder("[");
            for (double stat : stats)
            {
                result.append(stat).append(", ");
            }
            return result + "]";
        }

        public int getLength ()
        {
            return trainingResults.length;
        }

        @Override
        public String toString()
        {
            int data_length = this.trainingResults[0].wrong + this.trainingResults[0].correct;
            int test_length = this.testResults[0].wrong + this.testResults[0].correct;
            int iterations = this.trainingResults.length;
            StringBuilder detail = new StringBuilder();
            for (int i = 0; i < iterations; i++)
            {
                Result training_data = this.trainingResults[i];
                Result test_result = this.testResults[i];
                detail.append(i).append(". Training | correct: ").append(training_data.correct).append(" wrong: ").append(training_data.wrong).append(" absolute_error: ").append(arrayToString(training_data.absolute_output_error)).append(" relative_error: ").append(arrayToString(training_data.relative_output_error)).append("\n");
                detail.append(i).append(". Test | correct: ").append(test_result.correct).append(" wrong: ").append(test_result.wrong).append(" unclassified: ").append(test_result.unclassified).append(" absolute_error: ").append(arrayToString(test_result.absolute_output_error)).append(" relative_error: ").append(arrayToString(test_result.relative_output_error)).append("\n");
            }

            return "NeuralNetTrainingResult: Learned " + data_length + " datasets for " + iterations +
                    " times and tested with " + test_length + " datasets for " + iterations + " times in " + this.time +
                    " milliseconds:\n\n" + detail;

        }
    }

    /**
     * Konstruktor
     * @param input_size die maximale Größe des Inputs, die das KNN verarbeiten kann
     */
    public NeuralNet(int input_size)
    {
        this.inputSize = input_size;
    }

    protected int getInputSize ()
    {
        return inputSize;
    }

    protected ArrayList<HiddenLayer> getHiddenLayers()
    {
        return this.hiddenLayers;
    }


    public double getLearningRate()
    {
        return this.learningRate;
    }

    /**
     * Setzt den standartmäßig zu Layer gesetzten Bias Wert.
     * @param standardBias Wert
     */
    public void setStandardBias(double standardBias)
    {
        this.standardBias = standardBias;
    }

    protected void addLayer(double[][] weights, Layer.Function function, double bias)
    {
        this.addLayer(weights.length, function, 0, 0, bias);
        HiddenLayer layer = this.hiddenLayers.get(this.hiddenLayers.size() - 1);
        for (int i = 0; i < weights.length; i++)
        {
            layer.getNeuron(i).setWeights(weights[i]);
        }
    }

    protected void generateOutputLayer(double[][] weights, Layer.Function function, double bias)
    {
        this.generateOutputLayer(weights.length, function, 0, 0, bias);
        for (int i = 0; i < weights.length; i++)
        {
            this.outputLayer.getNeuron(i).setWeights(weights[i]);
        }
    }

    public void addLayer(int size, Layer.Function function)
    {
        this.addLayer(size, function, 0.3, -0.3);
    }

    public void addLayer(int size, Layer.Function function, double max_weight_value, double min_weight_value)
    {
        this.addLayer(size, function, max_weight_value, min_weight_value, this.standardBias);
    }

    /**
     * Fügt einen HiddenLayer zwischen dem OutputLayer und dem letzten HiddenLayer hinzu.
     * Achtung: Hierbei wird der OutputLayer vernichtet, so dass dieser dannach neu erstellt werden muss, damit das KNN startet
     * @param size Anzahl der Neuronen
     * @param function Aktiverungsfunktion
     * @param max_weight_value [optional] Maximalwert des Bereiches, in dem die Gewichte initialisiert werden.
     * @param min_weight_value [optional] Minimalwert des Bereiches, in dem die Gewichte initialisiert werden.
     * @param bias [optional] Der Bias des Layers (bleibt Konstant; nicht vollständig implementiert, daher experimentell)
     */
    public void addLayer(int size, Layer.Function function, double max_weight_value, double min_weight_value, double bias)
    {
        this.outputLayer = null;
        HiddenLayer layer = new HiddenLayer(this, this.hiddenLayers.size() + 1, size, function, null);
        if (this.hiddenLayers.size() > 0)
        {
            HiddenLayer before = this.hiddenLayers.get(this.hiddenLayers.size() - 1);
            layer.createNeurons(before.getSize(), min_weight_value, max_weight_value);
            before.setNextLayer(layer);
        } else
        {
            layer.createNeurons(this.inputSize, min_weight_value, max_weight_value);
        }
        layer.setBias(bias);

        this.hiddenLayers.add(layer);
    }

    public void generateOutputLayer(int size, Layer.Function function)
    {
        this.generateOutputLayer(size, function, 0.3, -0.3);
    }

    public void generateOutputLayer(int size, Layer.Function function, double max_weight_value, double min_weight_value)
    {
        this.generateOutputLayer(size, function, max_weight_value, min_weight_value, this.standardBias);
    }

    /**
     * Generiert den OutputLayer
     * Dies sollte erst passieren, nachdem alle HiddenLayer hinzugefügt wurden.
     * @param size Anzahl der Neuronen
     * @param function Aktiverungsfunktion
     * @param max_weight_value [optional] Maximalwert des Bereiches, in dem die Gewichte initialisiert werden.
     * @param min_weight_value [optional] Minimalwert des Bereiches, in dem die Gewichte initialisiert werden.
     * @param bias [optional] Der Bias des Layers (bleibt Konstant; nicht vollständig implementiert, daher experimentell)
     */
    public void generateOutputLayer(int size, Layer.Function function, double max_weight_value, double min_weight_value, double bias)
    {
        int hidden_layer_size = this.hiddenLayers.size();
        Layer last_hidden = hidden_layer_size != 0 ? this.hiddenLayers.get(hidden_layer_size - 1) : null;
        this.outputLayer = new OutputLayer(this, hidden_layer_size + 1, size, function);
        this.outputLayer.createNeurons(last_hidden == null ? this.inputSize : last_hidden.getSize(), min_weight_value, max_weight_value);
        this.outputLayer.setBias(bias);
        if (last_hidden != null)
        {
            last_hidden.setNextLayer(this.outputLayer);
        }
    }

    /**
     * Berechnet die Ausgabe zu einer Eingabe
     * @param inputVector Eingabe
     * @return Ausgabe
     */
    public double[] compute(double[] inputVector)
    {
        if (this.outputLayer == null)
        {
            System.out.println("Error: neural net used without an output layer");
            return new double[0];
        }
        double[] result_vector = inputVector;
        for (HiddenLayer hiddenLayer : this.hiddenLayers)
        {
            try
            {
                result_vector = hiddenLayer.compute(result_vector);
            } catch (TooManyInputsException e)
            {
                System.out.println("Critical Error while computing result: ");
                System.out.println(e.toString());
                return new double[0];
            }
        }

        try
        {
            return this.outputLayer.compute(result_vector);
        } catch (TooManyInputsException e)
        {
            System.out.println("Critical Error while computing result: ");
            System.out.println(e.toString());
            return new double[0];
        }
    }

    /**
     * Training anhand eines einzelnen Datensatzes
     * @param inputVector Eingabe
     * @param expectedVector Erwartete Ausgabe
     */
    public void train(double[] inputVector, double[] expectedVector)
    {
        if (this.compute(inputVector).length == 0) return;
        this.outputLayer.setExpectedOutput(expectedVector);
        this.outputLayer.train();
        for (int i = this.hiddenLayers.size() - 1; i >= 0; i--)
        {
            this.hiddenLayers.get(i).train();
        }
    }

    /**
     * Setzt den Toleranzwert des KNNs
     * @param tolerance Toleranzwert
     */
    public void setTolerance(double tolerance)
    {
        this.tolerance = tolerance;
    }

    public double getTolerance()
    {
        return tolerance;
    }

    /**
     * Setzt die Lernrate des KNNs
     * @param learningRate Lernrate
     */
    public void setLearningRate(double learningRate)
    {
        this.learningRate = learningRate;
    }

    /**
     * Trainiert in mehreren Zyklen anhand einer Liste von Trainings- und Testdatensätzen
     * Da dieser Vorgang sehr viel Zeit beanspruchen kann, werden nach jedem Zyklus Statusmeldungen auf der Konsole ausgegeben
     * @param train_data Der Input der Daten, mit denen trainiert werden soll
     * @param train_expectations Die erwartete Ausgabe der Daten, mit denen trainiert werden soll
     * @param test_data Der Input der Daten, mit denen getestet werden soll
     * @param test_expectations Die erwartete Ausgabe der Daten, mit denen getestet werden soll
     * @param iterations Anzahl der Zyklen
     * @return Datensatz mit Angaben zur Qualität der Ausgaben aus jedem Zyklus
     */
    public NeuralNetTrainingResult train(double[][] train_data, double[][] train_expectations, double[][] test_data, double[][] test_expectations, int iterations)
    {
        NeuralNetTrainingResult result = new NeuralNetTrainingResult();
        result.trainingResults = new NeuralNetTrainingResult.Result[iterations];
        result.testResults = new NeuralNetTrainingResult.Result[iterations];
        long millis = System.currentTimeMillis();

        for (int i = 0; i < iterations; i++)
        {
            //create error stats
            result.trainingResults[i] = new NeuralNetTrainingResult.Result();
            result.trainingResults[i].absolute_output_error = new double[this.outputLayer.getSize()];
            result.trainingResults[i].relative_output_error = new double[this.outputLayer.getSize()];
            result.testResults[i] = new NeuralNetTrainingResult.Result();
            result.testResults[i].absolute_output_error = new double[this.outputLayer.getSize()];
            result.testResults[i].relative_output_error = new double[this.outputLayer.getSize()];

            for (int j = 0; j < train_data.length; j++)
            {
                //train
                this.train(train_data[j], train_expectations[j]);

                //analyze error
                double[] error = this.outputLayer.getError();
                int highest_index = 0;
                double highest_value = Integer.MIN_VALUE;
                double sec_highest_value = Integer.MIN_VALUE;
                double[] output = outputLayer.getLastOutput();

                for (int k = 0; k < this.outputLayer.getSize(); k++)
                {
                    if (output[k] > highest_value)
                    {
                        highest_index = k;
                        sec_highest_value = highest_value;
                        highest_value = output[k];
                    } else if (sec_highest_value < output[k]) sec_highest_value = output[k];
                    result.trainingResults[i].relative_output_error[k] += error[k];
                    result.trainingResults[i].absolute_output_error[k] += Math.abs(error[k]);
                }

                int correct_label = 0;
                for (int k = 0; k < train_expectations[j].length; k++)
                {
                    if (train_expectations[j][k] == 1) {
                        correct_label = k;
                        break;
                    }
                }
                result.trainingResults[i].total_per_label[correct_label]++;
                if (highest_value < sec_highest_value + 0.2) result.trainingResults[i].unclassified++;
                else if (train_expectations[j][highest_index] != 1) result.trainingResults[i].wrong++;
                else
                {
                    result.trainingResults[i].correct++;
                    result.trainingResults[i].correct_per_label[correct_label]++;
                }
            }

            for (int j = 0; j < test_data.length; j++)
            {
                //compute result
                double[] output = this.compute(test_data[j]);
                //analyze error
                int highest_index = 0;
                double highest_value = Integer.MIN_VALUE;
                double sec_highest_value = Integer.MIN_VALUE;
                for (int k = 0; k < this.outputLayer.getSize(); k++)
                {
                    if (output[k] > highest_value)
                    {
                        highest_index = k;
                        sec_highest_value = highest_value;
                        highest_value = output[k];
                    } else if (sec_highest_value < output[k]) sec_highest_value = output[k];
                    double error = output[k] - test_expectations[j][k];
                    result.testResults[i].relative_output_error[k] += error;
                    result.testResults[i].absolute_output_error[k] += Math.abs(error);
                }
                int correct_label = 0;
                for (int k = 0; k < test_expectations[j].length; k++)
                {
                    if (test_expectations[j][k] == 1) {
                        correct_label = k;
                        break;
                    }
                }
                result.testResults[i].total_per_label[correct_label]++;
                if (highest_value < sec_highest_value + 0.2) result.testResults[i].unclassified++;
                else if (test_expectations[j][highest_index] != 1) result.testResults[i].wrong++;
                else
                {
                    result.testResults[i].correct++;
                    result.testResults[i].correct_per_label[correct_label]++;
                }
            }

            System.out.println("Round " + (i + 1) + " completed: " + result.testResults[i].correct + " / " + test_data.length + " correct");
        }

        result.time = (int) (System.currentTimeMillis() - millis);

        return result;
    }

    public OutputLayer getOutputLayer()
    {
        return outputLayer;
    }
}
