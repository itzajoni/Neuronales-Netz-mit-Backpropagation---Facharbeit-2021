/**
 * Copyright (c) 2021 Jonathan Dreisvogt, jdreisvogt.de
 * The source code(summarized as "Software") is property of jdreisvogt.de (Jonathan Dreisvogt)
 * The content of our software is subject to German copyright law.
 * You are allowed to copy, publish and use this software as long as you mention source and copyright
 */

package de.jdreisvogt.sandbox.neuralnet.test;

import javax.swing.*;
import java.awt.*;

/**
 * Klasse GUI
 * Repräsentiert eine einzelne Ausgabe eines Datensatzes inklusive berechnetem Ergebnis und Label
 */
public class GUI extends JFrame
{
    final JLabel[] results = new JLabel[10];
    final JPanel[][] pixel = new JPanel[28][28];

    /**
     * Konstruktor
     * Erstellt ein neues Fenster
     * @param correct_value korrektes Label des Datensatzes
     */
    public GUI(int correct_value)
    {
        super("Result");

        JLabel correct = new JLabel("Correct value: " + correct_value);
        correct.setBounds(100, 200, 100, 20);
        this.add(correct);
        int length = (200/28);
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                JPanel jPanel = new JPanel();
                jPanel.setBackground(new Color(255, 255, 255));
                jPanel.setBounds(20 + (i * length), 20 + (j * length), length, length);
                this.pixel[i][j] = jPanel;
                this.add(jPanel);
            }
        }
        for (int i = 0; i < 10; i++)
        {
            JLabel jLabel = new JLabel("0");
            jLabel.setBounds(260, 20 + (i * 22), 50, 21);
            this.add(jLabel);
            this.results[i] = jLabel;
            JTextPane jTextPane = new JTextPane();
            jTextPane.setText(i + ": ");
            jTextPane.setBounds(230, 20 + (i * 22), 28, 21);
            this.add(jTextPane);
        }
        this.add(new Label());
        this.setSize(500, 400);
        this.setLocationRelativeTo(null);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.setVisible(true);
    }

    /**
     * Setzt die Pixel, des Datensatzes. Diese werden in dem Fenster dargestellt
     * @param pixel zweidimenstionale Matrix mit Helligkeitswerten zwischen 0 (-> schwarz) und 1 (-> weis)
     */
    public void setPixel(double[][] pixel)
    {
        if (pixel.length != 28 || pixel[0].length != 28) return;
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                int color = (int) (pixel[j][i] * 255);
                this.pixel[i][j].setBackground(new Color(color, color, color));
            }
        }
    }

    /**
     * Setzt den Output des Neuronalen Netzes. Dieser wird auf dem Fenster angezeigt
     * Bei der Ausgabe wird vereinfacht von Prozentwerten ausgegangen, da das NN i. d. R. lernt, dass eine
     * Gesamtsumme der Outputs von 1 vorteilhaft ist. Diese Angabe ist technisch und mathematisch aber nicht korrekt.
     * @param results Ausgaben der Output-Neuronen (optimalerweise zwischen 0 und 1)
     */
    public void setResults(double[] results)
    {
        if (results.length != 10) return;
        for (int i = 0; i < 10; i++)
        {
            this.results[i].setText(((int) (results[i] * 100)) + "%");
        }
    }
}
