/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.layer;

import de.jdreisvogt.sandbox.neuralnet.net.NeuralNet;
import de.jdreisvogt.sandbox.neuralnet.neuron.Neuron;
import de.jdreisvogt.sandbox.neuralnet.neuron.OutputNeuron;
import de.jdreisvogt.sandbox.neuralnet.neuron.TooManyInputsException;

/**
 * Repräsentiert einen Hidden-Layer
 * Erbt von der Klasse Layer
 * Implementiert die Methode createNeurons(...), welche in diesem Fall OutputNeurons zurückgibt sowie weitere Funktionen (s.u.)
 */
public class OutputLayer extends Layer
{
    private final double[] lastDetectedError;

    public OutputLayer(NeuralNet parent, int id, int size, Function function)
    {
        super(parent, id, size, function, null);
        this.lastDetectedError = new double[size];
    }

    /**
     * Setzt die Ausgabe, die bei einem Training erwartet wurde
     * Sollte aufgerufen werden, nachdem die Trainingsdaten berechnet wurden
     * @param expected Die erwartete Ausgabe
     */
    public void setExpectedOutput(double[] expected)
    {
        if (expected.length != this.neurons.length) return;
        for (int i = 0; i < this.neurons.length; i++)
        {
            this.lastDetectedError[i] = this.lastOutput[i] - expected[i];
        }
    }

    @Override protected Neuron createNeuron(int max_input_size, int id, double min_weight_value, double max_weight_value)
    {
        return new OutputNeuron(this, max_input_size, id, min_weight_value, max_weight_value);
    }

    /**
     * Gibt den Fehler eines bestimmten Neurons zurück
     * @param neuron_id id des Neurons
     * @return Fehlerwert
     */
    public double getError(int neuron_id)
    {
        if (this.lastDetectedError.length <= neuron_id) return 0f;
        if (Math.abs(this.lastDetectedError[neuron_id]) <= this.parent.getTolerance()) return 0f;
        return this.lastDetectedError[neuron_id];
    }

    public double[] getError()
    {
        return this.lastDetectedError;
    }
}
