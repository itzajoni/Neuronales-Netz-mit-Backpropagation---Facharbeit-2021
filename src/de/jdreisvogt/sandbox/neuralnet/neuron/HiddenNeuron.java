/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.neuron;


import de.jdreisvogt.sandbox.neuralnet.layer.Layer;

/**
 * Repräsentiert ein Neuron in einem HiddenLayer
 * Implementiert den Backpropagation-Algorithmus für Hidden-Neuronen
 * Erbt von der Klasse Neuron
 */
public class HiddenNeuron extends Neuron
{

    public HiddenNeuron(Layer parent, int max_input_size, int neuron_id, double min_weight_value, double max_weight_value)
    {
        super(parent, max_input_size, neuron_id, min_weight_value, max_weight_value);
    }

    @Override
    public void changeWeights()
    {
        double derivative = this.layer.getFunction().computeDerivative(this.lastComputedInputSum);
        double sum_of_error_signals = 0;
        for (int i = 0; i < this.layer.getNextLayer().getSize(); i++)
        {
            Neuron neuron = this.layer.getNextLayer().getNeuron(i);
            sum_of_error_signals += neuron.lastComputedErrorSignal * neuron.weights[this.neuronId];
        }
        this.lastComputedErrorSignal = derivative * sum_of_error_signals;

        for (int i = 0; i < this.layer.getLastInput().length; i++)
        {
            this.weights[i] -= this.layer.getNet().getLearningRate() * this.lastComputedErrorSignal * this.layer.getLastInput()[i];
        }
    }

    @Override
    public double compute(double[] x) throws TooManyInputsException
    {
        return super.compute(x);
    }
}
