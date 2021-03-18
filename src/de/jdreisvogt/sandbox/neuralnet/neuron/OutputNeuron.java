/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.neuron;

import de.jdreisvogt.sandbox.neuralnet.layer.Layer;
import de.jdreisvogt.sandbox.neuralnet.layer.OutputLayer;
import de.jdreisvogt.sandbox.neuralnet.neuron.Neuron;

/**
 * Repräsentiert ein Neuron in einem OutputLayer
 * Implementiert den Backpropagation-Algorithmus für Output-Neuronen in changeWeights()
 * Erbt von der Klasse Neuron
 */
public class OutputNeuron extends Neuron
{
    public OutputNeuron(Layer parent, int max_input_size, int neuron_id, double min_weight_value, double max_weight_value)
    {
        super(parent, max_input_size, neuron_id, min_weight_value, max_weight_value);
    }

    @Override
    public void changeWeights()
    {
        if (!(this.layer instanceof OutputLayer)) return;
        OutputLayer layer = (OutputLayer) this.layer;

        double derivative = this.layer.getFunction().computeDerivative(this.lastComputedInputSum);
        this.lastComputedErrorSignal = layer.getError(this.neuronId) * derivative;
        for (int i = 0; i < this.layer.getLastInput().length; i++)
        {
            this.weights[i] -= this.layer.getNet().getLearningRate() * this.lastComputedErrorSignal * this.layer.getLastInput()[i];
        }
    }
}
