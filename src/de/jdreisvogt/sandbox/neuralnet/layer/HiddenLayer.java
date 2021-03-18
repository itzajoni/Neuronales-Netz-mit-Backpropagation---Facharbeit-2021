/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 *
 *  Any misuse of the content is prohibited and will be prosecuted by
 *  GamingGeneration.de under civil and criminal law.
 */

package de.jdreisvogt.sandbox.neuralnet.layer;

import de.jdreisvogt.sandbox.neuralnet.net.NeuralNet;
import de.jdreisvogt.sandbox.neuralnet.neuron.HiddenNeuron;
import de.jdreisvogt.sandbox.neuralnet.neuron.Neuron;

/**
 * Repräsentiert einen Hidden-Layer
 * Erbt von der Klasse Layer
 * Implementiert die Methode createNeurons(...), welche in diesem Fall HiddenNeurons zurückgibt
 */
public class HiddenLayer extends Layer
{
    public HiddenLayer(NeuralNet parent, int id, int size, Function function, Layer nextLayer)
    {
        super(parent, id, size, function, nextLayer);
    }

    @Override
    protected Neuron createNeuron(int max_input_size, int id, double min_weight_value, double max_weight_value)
    {
        return new HiddenNeuron(this, max_input_size, id, min_weight_value, max_weight_value);
    }
}
