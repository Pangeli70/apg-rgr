/** -----------------------------------------------------------------------
 * @module [REGR]
 * @author [APG] ANGELI Paolo Giusto
 * @version 0.8.0 [APG 2022/05/29] Porting to Deno 
 * @credits https://github.com/Tom-Alexander/regression-js/blob/master/src/regression.js
 * -----------------------------------------------------------------------
 */
import { IApg2DPoint } from "../../../2D/mod.ts"
import { eApgRegrTypes } from "../enums/eApgRegrTypes.ts";
import { IApgRegrOptions } from "../interfaces/IApgRegrOptions.ts";
import { IApgRegrResult } from "../interfaces/IApgRegrResult.ts";

export class ApgRegrCalc {

  readonly DEFAULT_OPTIONS: IApgRegrOptions = {
    order: 2,
    precision: 3
  };

  /**
  * Determine the coefficient of determination (r^2) of a fit from the observations
  * and predictions.
  *
  * @param {Array<Array<number>>} apoints - Pairs of observed x-y values
  * @param {Array<Array<number>>} results - Pairs of observed predicted x-y values
  *
  * @return {number} - The r^2 value, or NaN if one cannot be calculated.
  */
  determinationCoefficient(apoints: IApg2DPoint[], results: IApg2DPoint[]) {
    const predictions: IApg2DPoint[] = [];
    const observations: IApg2DPoint[] = [];

    apoints.forEach((point, i) => {
      if (point.y !== null) {
        observations.push(point);
        predictions.push(results[i]);
      }
    });

    const sum = observations.reduce((accum, acurrent) => accum + acurrent.y, 0);
    const mean = sum / observations.length;

    const squaredDeviationFromMean = observations.reduce((accum, acurrent) => {
      const difference = acurrent.y - mean;
      return accum + (difference * difference);
    }, 0);

    const squardDifferenceFromPredicted = observations.reduce((accum, acurrent, index) => {
      const prediction = predictions[index];
      const residual = acurrent.y - prediction.y;
      return accum + (residual * residual);
    }, 0);

    return 1 - (squardDifferenceFromPredicted / squaredDeviationFromMean);
  }

  /**
  * Determine the solution of a system of linear equations A * x = b using
  * Gaussian elimination.
  *
  * @param input A 2-d matrix of data in row-major form [ A | b ]
  * @param order How many degrees to solve for
  *
  * @return Vector of normalized solution coefficients matrix (x)
  */
  private gaussianElimination(input: number[][], order: number) {
    const matrix = input;
    const n = input.length - 1;
    const coefficients = [order];

    for (let i = 0; i < n; i++) {
      let maxrow = i;
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(matrix[i][j]) > Math.abs(matrix[i][maxrow])) {
          maxrow = j;
        }
      }

      for (let k = i; k < n + 1; k++) {
        const tmp = matrix[k][i];
        matrix[k][i] = matrix[k][maxrow];
        matrix[k][maxrow] = tmp;
      }

      for (let j = i + 1; j < n; j++) {
        for (let k = n; k >= i; k--) {
          matrix[k][j] -= (matrix[k][i] * matrix[i][j]) / matrix[i][i];
        }
      }
    }

    for (let j = n - 1; j >= 0; j--) {
      let total = 0;
      for (let k = j + 1; k < n; k++) {
        total += matrix[k][j] * coefficients[k];
      }

      coefficients[j] = (matrix[n][j] - total) / matrix[j][j];
    }

    return coefficients;
  }

  /**
  * Round a number to a precision, specificed in number of decimal places
  *
  * @param number The number to round
  * @param precision The number of decimal places to round to:
  *                  > 0 means decimals, < 0 means powers of 10
  * @return The number, rounded
  */
  private round(number: number, precision: number) {
    const factor = 10 ** precision;
    return Math.round(number * factor) / factor;
  }



  linear(apoints: IApg2DPoint[], aoptions: IApgRegrOptions) {
    const sum = [0, 0, 0, 0, 0];
    let len = 0;

    for (let n = 0; n < apoints.length; n++) {
      if (apoints[n].y !== null) {
        len++;
        sum[0] += apoints[n].x;
        sum[1] += apoints[n].y;
        sum[2] += apoints[n].x * apoints[n].x;
        sum[3] += apoints[n].x * apoints[n].y;
        sum[4] += apoints[n].y * apoints[n].y;
      }
    }

    const run = ((len * sum[2]) - (sum[0] * sum[0]));
    const rise = ((len * sum[3]) - (sum[0] * sum[1]));
    const gradient = run === 0 ? 0 : this.round(rise / run, aoptions.precision);
    const intercept = this.round((sum[1] / len) - ((gradient * sum[0]) / len), aoptions.precision);

    const predictLinearFn = (x: number) => {
      const newY = (gradient * x) + intercept;
      const r: IApg2DPoint = {
        x: this.round(x, aoptions.precision),
        y: this.round(newY, aoptions.precision)
      }
      return r;
    }

    const predicted = apoints.map(point => predictLinearFn(point.x));
    const r2 = this.determinationCoefficient(apoints, predicted);

    const r: IApgRegrResult = {
      points: apoints,
      predicted: predicted,
      type: eApgRegrTypes.LINEAR,
      coefficients: [gradient, intercept],
      equation: intercept === 0 ? `y = ${gradient}x` : `y = ${gradient}x + ${intercept}`,
      r2: this.round(r2, aoptions.precision),
    };
    return r;
  }

  exponential(apoints: IApg2DPoint[], aoptions: IApgRegrOptions) {
    const sum = [0, 0, 0, 0, 0, 0];

    for (let n = 0; n < apoints.length; n++) {
      if (apoints[n].y !== null) {
        sum[0] += apoints[n].x;
        sum[1] += apoints[n].y;
        sum[2] += apoints[n].x * apoints[n].x * apoints[n].y;
        sum[3] += apoints[n].y * Math.log(apoints[n].y);
        sum[4] += apoints[n].x * apoints[n].y * Math.log(apoints[n].y);
        sum[5] += apoints[n].x * apoints[n].y;
      }
    }

    const denominator = ((sum[1] * sum[2]) - (sum[5] * sum[5]));
    const a = Math.exp(((sum[2] * sum[3]) - (sum[5] * sum[4])) / denominator);
    const b = ((sum[1] * sum[4]) - (sum[5] * sum[3])) / denominator;
    const coeffA = this.round(a, aoptions.precision);
    const coeffB = this.round(b, aoptions.precision);


    const predictExponentialFn = (x: number) => {
      const newY = coeffA * Math.exp(coeffB * x);
      const r: IApg2DPoint = {
        x: this.round(x, aoptions.precision),
        y: this.round(newY, aoptions.precision)
      }
      return r;
    }

    const predicted = apoints.map(point => predictExponentialFn(point.x));
    const r2 = this.determinationCoefficient(apoints, predicted);

    const r: IApgRegrResult = {
      points: apoints,
      predicted: predicted,
      type: eApgRegrTypes.EXPONENTIAL,
      coefficients: [coeffA, coeffB],
      equation: `y = ${coeffA}e^(${coeffB}x)`,
      r2: this.round(r2, aoptions.precision)
    };
    return r;
  }

  logarithmic(apoints: IApg2DPoint[], aoptions: IApgRegrOptions) {
    const sum = [0, 0, 0, 0];
    const len = apoints.length;

    for (let n = 0; n < len; n++) {
      if (apoints[n].y !== null) {
        sum[0] += Math.log(apoints[n].x);
        sum[1] += apoints[n].y * Math.log(apoints[n].x);
        sum[2] += apoints[n].y;
        sum[3] += (Math.log(apoints[n].x) ** 2);
      }
    }

    const a = ((len * sum[1]) - (sum[2] * sum[0])) / ((len * sum[3]) - (sum[0] * sum[0]));
    const coeffB = this.round(a, aoptions.precision);
    const coeffA = this.round((sum[2] - (coeffB * sum[0])) / len, aoptions.precision);

    const predictLogaritmicFn = (x: number) => {
      const newY = coeffA + (coeffB * Math.log(x));
      const r: IApg2DPoint = {
        x: this.round(x, aoptions.precision),
        y: this.round(newY, aoptions.precision)
      }
      return r;
    }


    const predicted = apoints.map(point => predictLogaritmicFn(point.x));
    const r2 = this.determinationCoefficient(apoints, predicted);

    const r: IApgRegrResult = {
      points: apoints,
      predicted: predicted,
      type: eApgRegrTypes.LOGARITMIC,
      coefficients: [coeffA, coeffB],
      equation: `y = ${coeffA} + ${coeffB} ln(x)`,
      r2: this.round(r2, aoptions.precision)
    };
    return r;
  }

  power(apoints: IApg2DPoint[], aoptions: IApgRegrOptions) {
    const sum = [0, 0, 0, 0, 0];
    const len = apoints.length;

    for (let n = 0; n < len; n++) {
      if (apoints[n].y !== null) {
        sum[0] += Math.log(apoints[n].x);
        sum[1] += Math.log(apoints[n].y) * Math.log(apoints[n].x);
        sum[2] += Math.log(apoints[n].y);
        sum[3] += (Math.log(apoints[n].x) ** 2);
      }
    }

    const b = ((len * sum[1]) - (sum[0] * sum[2])) / ((len * sum[3]) - (sum[0] ** 2));
    const a = ((sum[2] - (b * sum[0])) / len);
    const coeffA = this.round(Math.exp(a), aoptions.precision);
    const coeffB = this.round(b, aoptions.precision);

    const predictPowerFn = (ax: number) => {
      const newY = coeffA * (ax ** coeffB);
      const r: IApg2DPoint = {
        x: this.round(ax, aoptions.precision),
        y: this.round(newY, aoptions.precision)
      }
      return r;
    }

    const predicted = apoints.map(point => predictPowerFn(point.x));
    const r2 = this.determinationCoefficient(apoints, predicted);
    const r: IApgRegrResult = {
      points: apoints,
      predicted: predicted,
      type: eApgRegrTypes.POWER,
      coefficients: [coeffA, coeffB],
      equation: `y = ${coeffA}x^${coeffB}`,
      r2: this.round(r2, aoptions.precision),
    };
    return r;
  }

  polynomial(apoints: IApg2DPoint[], aoptions: IApgRegrOptions) {
    const lhs = [];
    const rhs = [];
    let a = 0;
    let b = 0;
    const len = apoints.length;
    const k = aoptions.order + 1;

    for (let i = 0; i < k; i++) {
      for (let l = 0; l < len; l++) {
        if (apoints[l].y !== null) {
          a += (apoints[l].x ** i) * apoints[l].y;
        }
      }

      lhs.push(a);
      a = 0;

      const c = [];
      for (let j = 0; j < k; j++) {
        for (let l = 0; l < len; l++) {
          if (apoints[l].y !== null) {
            b += apoints[l].x ** (i + j);
          }
        }
        c.push(b);
        b = 0;
      }
      rhs.push(c);
    }
    rhs.push(lhs);

    const coefficients =
      this.gaussianElimination(rhs, k)
        .map(v => this.round(v, aoptions.precision));

    const predictPolynomialFn = (x: number) => {
      const newY = coefficients.reduce((sum, coeff, power) => sum + (coeff * (x ** power)), 0);
      const r: IApg2DPoint = {
        x: this.round(x, aoptions.precision),
        y: this.round(newY, aoptions.precision)
      }
      return r;
    };

    const predicted = apoints.map((apoint: IApg2DPoint) => predictPolynomialFn(apoint.x));

    let equation = 'y = ';
    for (let i = coefficients.length - 1; i >= 0; i--) {
      if (i > 1) {
        equation += `${coefficients[i]}x^${i} + `;
      } else if (i === 1) {
        equation += `${coefficients[i]}x + `;
      } else {
        equation += coefficients[i];
      }
    }

    const r2 = this.determinationCoefficient(apoints, predicted);

    const r: IApgRegrResult = {
      points: apoints,
      predicted: predicted,
      type: eApgRegrTypes.POLYNOMIAL,
      coefficients: [...coefficients].reverse(),
      equation,
      r2: this.round(r2, aoptions.precision)
    };
    return r;
  }
}


