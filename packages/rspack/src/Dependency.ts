import { type JsDependency, JsDependencyMut } from "@rspack/binding";

export class Dependency {
	declare readonly type: string;
	declare readonly category: string;
	declare readonly request: string | undefined;
	declare critical: boolean;

	static __from_binding(binding: JsDependencyMut | JsDependency): Dependency {
		return new Dependency(binding);
	}

	private constructor(binding: JsDependencyMut | JsDependency) {
		Object.defineProperties(this, {
			type: {
				enumerable: true,
				get(): string {
					return binding.type;
				}
			},
			category: {
				enumerable: true,
				get(): string {
					return binding.category;
				}
			},
			request: {
				enumerable: true,
				get(): string | undefined {
					return binding.request;
				}
			},
			critical: {
				enumerable: true,
				get(): boolean {
					return binding.critical;
				},
				set(val: boolean) {
					if (binding instanceof JsDependencyMut) {
						binding.critical = val;
					}
				}
			}
		});
	}
}