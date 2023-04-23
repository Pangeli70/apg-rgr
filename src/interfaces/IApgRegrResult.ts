/** -----------------------------------------------------------------------
 * @module [REGR]
 * @author [APG] ANGELI Paolo Giusto
 * @version 0.8.0 [APG 2022/05/29] Porting to Deno 
 * -----------------------------------------------------------------------
 */
import { IApg2DPoint } from "../../../2D/mod.ts";
import { eApgRegrTypes } from "../enums/eApgRegrTypes.ts";


export interface IApgRegrResult {
  points: IApg2DPoint[];
  predicted: IApg2DPoint[];
  type: eApgRegrTypes;
  coefficients: number[];
  equation: string;
  r2: number;
}
