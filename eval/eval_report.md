# Evaluation Report

## Overall
| Metric                     | Value   |
|----------------------------|---------|
| Total questions            | 44      |
| Retrieval precision (avg)  | 15.00%  |
| Retrieval recall (avg)     | 61.36%  |
| Answer correctness         | 68.18%  |
| Avg latency (sec/question) | 10.15   |
| Correct abstention rate    | 100.00% |

## By question type
| question_type   |   n | retrieval_recall   | answer_correct   |
|-----------------|-----|--------------------|------------------|
| factual         |  17 | 53%                | 59%              |
| not_answerable  |   5 | 0%                 | 100%             |
| numeric         |  10 | 80%                | 80%              |
| table           |  12 | 83%                | 58%              |

## Per-question detail
| question                                                                                       | question_type   |   retrieval_recall | answer_correct   |   total_sec |
|------------------------------------------------------------------------------------------------|-----------------|--------------------|------------------|-------------|
| What is the maximum supply voltage for the UNO SPE Shield?                                     | table           |                  1 | True             |      11.538 |
| What RS-485 transceiver IC does the UNO SPE Shield use?                                        | factual         |                  0 | True             |       8.504 |
| What is the maximum operating current of the UNO SPE Shield?                                   | table           |                  0 | False            |       5.075 |
| What is the maximum bus length for the 10BASE-T1S SPE network on the UNO SPE Shield?           | numeric         |                  1 | True             |       5.48  |
| How many nodes can the UNO SPE Shield support in a multidrop SPE network?                      | numeric         |                  1 | True             |       5.884 |
| What ingress protection or environmental rating does the UNO SPE Shield housing have?          | not_answerable  |                  0 | True             |       3.439 |
| What is the maximum operating current the GPIOs on the Nano ESP32 can source?                  | numeric         |                  1 | True             |       3.933 |
| What is the recommended VIN input voltage range for the Nano ESP32?                            | table           |                  1 | True             |       3.964 |
| What is the maximum ambient operating temperature for the Nano ESP32?                          | table           |                  1 | True             |       4.904 |
| What radio module is the Nano ESP32 based on?                                                  | factual         |                  0 | True             |       5.754 |
| How much external flash memory does the Nano ESP32 have?                                       | numeric         |                  1 | True             |       8.042 |
| What is the operating voltage of the Nano ESP32?                                               | factual         |                  1 | True             |      10.048 |
| What Bluetooth version does the Nano ESP32 support?                                            | factual         |                  0 | False            |       9.305 |
| What are the default I2C pins on the Nano ESP32?                                               | factual         |                  1 | True             |       9.652 |
| How much PSRAM does the Nano ESP32's radio module include?                                     | numeric         |                  1 | True             |       9.904 |
| Does the Nano ESP32 support native Ethernet connectivity?                                      | not_answerable  |                  0 | True             |       9.766 |
| What microprocessor (MPU) does the Arduino UNO Q use?                                          | factual         |                  0 | False            |      15.09  |
| What real-time microcontroller (MCU) does the Arduino UNO Q use?                               | factual         |                  0 | False            |      11.464 |
| What is the maximum current the UNO Q can draw over USB-C VBUS?                                | table           |                  1 | True             |      14.547 |
| What is the recommended DC input voltage range for the UNO Q's VIN?                            | table           |                  1 | False            |      15.095 |
| What is the maximum operating temperature for the Arduino UNO Q?                               | table           |                  1 | True             |       8.934 |
| What are the two RAM/storage configuration options for the UNO Q?                              | factual         |                  1 | True             |      10.825 |
| What operating system does the MPU on the UNO Q run?                                           | factual         |                  1 | True             |      13.258 |
| What is the tolerance range of the UNO Q's 3.3 V system rail?                                  | table           |                  1 | True             |      16.861 |
| How many pins does the JMEDIA connector on the UNO Q have and at what voltage does it operate? | table           |                  1 | False            |      15.029 |
| What Wi-Fi standard does the UNO Q support?                                                    | factual         |                  0 | True             |      12.944 |
| What maximum display resolution does the UNO Q support in SBC mode?                            | factual         |                  1 | True             |      15.284 |
| How much RAM does the STM32U585 microcontroller on the UNO Q have?                             | numeric         |                  0 | False            |      13.403 |
| Can the UNO Q output both USB-C DisplayPort and JMEDIA MIPI-DSI video simultaneously?          | factual         |                  1 | False            |      15.366 |
| What is the maximum GPS accuracy of the Arduino UNO Q?                                         | not_answerable  |                  0 | True             |       7.794 |
| Which Arduino board in this set supports Single Pair Ethernet, and which supports Wi-Fi 5?     | factual         |                  1 | False            |       9.715 |
| What microcontroller does the Arduino UNO R3 use as its main processor?                        | factual         |                  1 | True             |      11.373 |
| What is the maximum input voltage from the VIN pad on the UNO R3?                              | table           |                  1 | False            |       9.446 |
| What is the minimum input voltage from the VIN pad on the UNO R3?                              | table           |                  1 | True             |       9.498 |
| What is the maximum operating temperature for the UNO R3?                                      | table           |                  0 | False            |      10.634 |
| How much SRAM does the ATmega328P on the UNO R3 have?                                          | numeric         |                  0 | True             |      15.087 |
| How much flash memory does the ATmega328P on the UNO R3 have?                                  | numeric         |                  1 | True             |       8.465 |
| What secondary processor handles USB communication on the UNO R3?                              | factual         |                  1 | True             |      11.61  |
| What is the maximum current draw of the ATMEGA328P-PU according to the power tree?             | numeric         |                  1 | False            |       9.182 |
| What connector type is used to power the UNO R3 via the barrel jack?                           | factual         |                  0 | False            |       7.353 |
| How many digital I/O pins does the UNO R3 have?                                                | numeric         |                  1 | True             |      11.913 |
| What is the maximum power consumption (in mA) of the UNO R3?                                   | not_answerable  |                  0 | True             |       9.547 |
| Does the UNO R3 have built-in Wi-Fi?                                                           | not_answerable  |                  0 | True             |      11.791 |
| Between the UNO R3 and the Nano ESP32, which board has wireless connectivity built in?         | factual         |                  0 | False            |       9.785 |